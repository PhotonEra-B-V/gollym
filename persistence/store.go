package persistence

import (
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"sync"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"

	"github.com/PhotonEra-B-V/gollym/core"
)

//go:embed schema.sql
var schemaSQL string

// DefaultTablePrefix is pyllym's default table_prefix.
const DefaultTablePrefix = "llm_"

// ErrNotFound is returned when a chat doesn't exist.
var ErrNotFound = errors.New("persistence: chat not found")

// DBTX is what the store needs from pgx: *pgxpool.Pool, *pgx.Conn and
// pgx.Tx all satisfy it. Pass a transaction to get pyllym's session
// semantics (writes become visible when you commit).
type DBTX interface {
	Exec(ctx context.Context, sql string, args ...any) (pgconn.CommandTag, error)
	Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error)
	QueryRow(ctx context.Context, sql string, args ...any) pgx.Row
}

// Store reads and writes chats, messages and tool calls.
type Store struct {
	db     DBTX
	prefix string
}

// Option configures New.
type Option func(*Store)

// WithTablePrefix names the tables {prefix}chats, {prefix}messages and
// {prefix}tool_calls (default llm_).
func WithTablePrefix(prefix string) Option { return func(s *Store) { s.prefix = prefix } }

var identRE = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

// New returns a store over db.
func New(db DBTX, opts ...Option) (*Store, error) {
	s := &Store{db: db, prefix: DefaultTablePrefix}
	for _, o := range opts {
		o(s)
	}
	if s.prefix != "" && !identRE.MatchString(s.prefix) {
		return nil, fmt.Errorf("persistence: invalid table prefix %q", s.prefix)
	}
	return s, nil
}

// WithDB returns a store with the same tables over another connection or
// transaction.
func (s *Store) WithDB(db DBTX) *Store { return &Store{db: db, prefix: s.prefix} }

// Schema returns the DDL for the given table prefix, for use in migrations.
func Schema(prefix string) (string, error) {
	if prefix != "" && !identRE.MatchString(prefix) {
		return "", fmt.Errorf("persistence: invalid table prefix %q", prefix)
	}
	return strings.ReplaceAll(schemaSQL, "{{prefix}}", prefix), nil
}

// Schema returns this store's DDL.
func (s *Store) Schema() string {
	ddl, _ := Schema(s.prefix)
	return ddl
}

// CreateTables creates the tables if they don't exist (metadata.create_all).
func (s *Store) CreateTables(ctx context.Context) error {
	_, err := s.db.Exec(ctx, s.Schema())
	return err
}

func (s *Store) table(name string) string { return s.prefix + name }

// CreateChat inserts a chat row and sets c.ID.
func (s *Store) CreateChat(ctx context.Context, c *Chat) error {
	q := fmt.Sprintf(`INSERT INTO %s (model_id, provider) VALUES ($1, $2) RETURNING id`, s.table("chats"))
	return s.db.QueryRow(ctx, q, c.ModelID, c.Provider).Scan(&c.ID)
}

// DeleteChat deletes a chat; its messages and tool calls cascade.
func (s *Store) DeleteChat(ctx context.Context, id int64) error {
	_, err := s.db.Exec(ctx, fmt.Sprintf(`DELETE FROM %s WHERE id = $1`, s.table("chats")), id)
	return err
}

// InsertMessage inserts a message and its tool calls, setting their IDs.
func (s *Store) InsertMessage(ctx context.Context, m *Message) error {
	citations, err := jsonOrNil(m.Citations)
	if err != nil {
		return err
	}
	q := fmt.Sprintf(`INSERT INTO %s (chat_id, role, content, model_id, tool_call_id, input_tokens,
		output_tokens, cached_tokens, cache_creation_tokens, thinking_tokens, thinking_text,
		thinking_signature, citations, finish_reason)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14) RETURNING id`, s.table("messages"))
	if err := s.db.QueryRow(ctx, q, m.ChatID, m.Role, m.Content, m.ModelID, m.ToolCallID, m.InputTokens,
		m.OutputTokens, m.CachedTokens, m.CacheCreationTokens, m.ThinkingTokens, m.ThinkingText,
		m.ThinkingSignature, citations, m.FinishReason).Scan(&m.ID); err != nil {
		return err
	}
	tq := fmt.Sprintf(`INSERT INTO %s (message_id, tool_call_id, name, arguments) VALUES ($1, $2, $3, $4) RETURNING id`, s.table("tool_calls"))
	for _, tc := range m.ToolCalls {
		args := tc.Arguments
		if args == nil {
			args = map[string]any{} // column default (dict)
		}
		raw, err := json.Marshal(args)
		if err != nil {
			return err
		}
		id := m.ID
		tc.MessageID = &id
		if err := s.db.QueryRow(ctx, tq, id, tc.ToolCallID, tc.Name, raw).Scan(&tc.ID); err != nil {
			return err
		}
	}
	return nil
}

func jsonOrNil(c []core.Citation) ([]byte, error) {
	if len(c) == 0 {
		return nil, nil
	}
	return json.Marshal(c)
}

// LoadChat reads a chat with its messages (by id) and their tool calls.
func (s *Store) LoadChat(ctx context.Context, id int64) (*Chat, error) {
	c := &Chat{ID: id}
	q := fmt.Sprintf(`SELECT model_id, provider FROM %s WHERE id = $1`, s.table("chats"))
	if err := s.db.QueryRow(ctx, q, id).Scan(&c.ModelID, &c.Provider); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, fmt.Errorf("%w: %d", ErrNotFound, id)
		}
		return nil, err
	}
	msgs, err := s.Messages(ctx, id)
	if err != nil {
		return nil, err
	}
	c.Messages = msgs
	return c, nil
}

// Messages reads a chat's messages ordered by id, with their tool calls.
func (s *Store) Messages(ctx context.Context, chatID int64) ([]*Message, error) {
	q := fmt.Sprintf(`SELECT id, chat_id, role, content, model_id, tool_call_id, input_tokens,
		output_tokens, cached_tokens, cache_creation_tokens, thinking_tokens, thinking_text,
		thinking_signature, citations, finish_reason FROM %s WHERE chat_id = $1 ORDER BY id`, s.table("messages"))
	rows, err := s.db.Query(ctx, q, chatID)
	if err != nil {
		return nil, err
	}
	var (
		msgs []*Message
		ids  []int64
		byID = map[int64]*Message{}
	)
	for rows.Next() {
		m := &Message{}
		var citations []byte
		if err := rows.Scan(&m.ID, &m.ChatID, &m.Role, &m.Content, &m.ModelID, &m.ToolCallID, &m.InputTokens,
			&m.OutputTokens, &m.CachedTokens, &m.CacheCreationTokens, &m.ThinkingTokens, &m.ThinkingText,
			&m.ThinkingSignature, &citations, &m.FinishReason); err != nil {
			rows.Close()
			return nil, err
		}
		// SQLAlchemy writes None as JSON null; both mean "no citations".
		if len(citations) > 0 && string(citations) != "null" {
			if err := json.Unmarshal(citations, &m.Citations); err != nil {
				rows.Close()
				return nil, fmt.Errorf("persistence: message %d citations: %w", m.ID, err)
			}
		}
		msgs = append(msgs, m)
		ids = append(ids, m.ID)
		byID[m.ID] = m
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if len(ids) == 0 {
		return msgs, nil
	}
	tq := fmt.Sprintf(`SELECT id, message_id, tool_call_id, name, arguments FROM %s
		WHERE message_id = ANY($1) ORDER BY id`, s.table("tool_calls"))
	trows, err := s.db.Query(ctx, tq, ids)
	if err != nil {
		return nil, err
	}
	defer trows.Close()
	for trows.Next() {
		tc := &ToolCall{}
		var args []byte
		if err := trows.Scan(&tc.ID, &tc.MessageID, &tc.ToolCallID, &tc.Name, &args); err != nil {
			return nil, err
		}
		if len(args) > 0 {
			if err := json.Unmarshal(args, &tc.Arguments); err != nil {
				return nil, fmt.Errorf("persistence: tool call %d arguments: %w", tc.ID, err)
			}
		}
		if m := byID[*tc.MessageID]; m != nil {
			m.ToolCalls = append(m.ToolCalls, tc)
		}
	}
	return msgs, trows.Err()
}

// ChatOptions configure ToChat.
type ChatOptions struct {
	AssumeModelExists bool
	Context           *core.Context
}

// ToChat returns a core.Chat backed by rec: its loaded messages become the
// history, and every message the chat adds afterwards (the staged user
// message, tool results, replies) is inserted when the chat's
// after-message callback fires. With a transaction as the store's DB you
// still commit yourself.
func (s *Store) ToChat(rec *Chat, opts ChatOptions) (*core.Chat, error) {
	chat, err := core.NewChat(core.ChatOptions{
		Model:             rec.ModelID,
		Provider:          core.Deref(rec.Provider),
		AssumeModelExists: opts.AssumeModelExists,
		Context:           opts.Context,
	})
	if err != nil {
		return nil, err
	}
	history := make([]*core.Message, 0, len(rec.Messages))
	for _, m := range rec.Messages {
		msg, err := m.ToLLM()
		if err != nil {
			return nil, err
		}
		history = append(history, msg)
	}
	chat.Messages = history

	var mu sync.Mutex
	persisted := make(map[*core.Message]bool, len(history))
	for _, m := range history {
		persisted[m] = true
	}
	// Every not-yet-saved message is written, so the staged user message
	// goes in alongside the first assistant reply.
	chat.AfterMessage(func(ctx context.Context, _ *core.Message) error {
		mu.Lock()
		defer mu.Unlock()
		for _, m := range chat.Messages {
			if persisted[m] {
				continue
			}
			if err := s.InsertMessage(ctx, MessageFromLLM(m, rec.ID)); err != nil {
				return fmt.Errorf("persistence: saving %s message: %w", m.Role, err)
			}
			persisted[m] = true
		}
		return nil
	})
	return chat, nil
}
