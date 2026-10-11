-- gollym persistence schema: the tables pyllym.persistence.create_models
-- builds with SQLAlchemy (same names, columns and types). {{prefix}} is the
-- table prefix (default llm_). The two indexes are additions: SQLAlchemy
-- doesn't index foreign keys, and LoadChat reads through them.

CREATE TABLE IF NOT EXISTS {{prefix}}chats (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(255) NOT NULL,
    provider VARCHAR(64)
);

CREATE TABLE IF NOT EXISTS {{prefix}}messages (
    id SERIAL PRIMARY KEY,
    chat_id INTEGER NOT NULL REFERENCES {{prefix}}chats (id) ON DELETE CASCADE,
    role VARCHAR(32) NOT NULL,
    content TEXT,
    model_id VARCHAR(255),
    tool_call_id VARCHAR(255),
    input_tokens INTEGER,
    output_tokens INTEGER,
    cached_tokens INTEGER,
    cache_creation_tokens INTEGER,
    thinking_tokens INTEGER,
    thinking_text TEXT,
    thinking_signature TEXT,
    citations JSON,
    finish_reason VARCHAR(64)
);

CREATE INDEX IF NOT EXISTS {{prefix}}messages_chat_id_idx ON {{prefix}}messages (chat_id);

CREATE TABLE IF NOT EXISTS {{prefix}}tool_calls (
    id SERIAL PRIMARY KEY,
    message_id INTEGER REFERENCES {{prefix}}messages (id) ON DELETE CASCADE,
    tool_call_id VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    arguments JSON NOT NULL
);

CREATE INDEX IF NOT EXISTS {{prefix}}tool_calls_message_id_idx ON {{prefix}}tool_calls (message_id);
