// Package riverjobs runs gollym operations as River background jobs
// (pyllym.celery). Register adds workers for the same task set as pyllym's
// create_tasks: ask, embed, paint, speak, transcribe and moderate.
//
//	workers := river.NewWorkers()
//	tasks, err := gollymriver.Register(workers) // tasks.Ask(...) runs in-process
//	client, _ := river.NewClient(riverpgxv5.New(pool), &river.Config{Workers: workers, ...})
//	_, _ = client.Insert(ctx, gollymriver.AskArgs{Prompt: "What's the capital of France?", Model: "gpt-5.4"}, nil)
//
// Arguments are plain JSON values (as with a Celery broker). Each job's
// result, the same dict pyllym's task returns, is stored with
// river.RecordOutput in the job's metadata under "output". Queue, retries
// and priority are River insert options, set when inserting.
package riverjobs

import (
	"context"
	"encoding/base64"
	"fmt"

	"github.com/riverqueue/river"

	"github.com/PhotonEra-B-V/gollym/core"
)

// Job kinds (pyllym's task names with a gollym prefix).
const (
	KindAsk        = "gollym.ask"
	KindEmbed      = "gollym.embed"
	KindPaint      = "gollym.paint"
	KindSpeak      = "gollym.speak"
	KindTranscribe = "gollym.transcribe"
	KindModerate   = "gollym.moderate"
)

// Kinds lists every job kind Register adds.
func Kinds() []string {
	return []string{KindAsk, KindEmbed, KindPaint, KindSpeak, KindTranscribe, KindModerate}
}

// Option configures Register.
type Option func(*Tasks)

// WithContext runs every job in a gollym configuration scope instead of
// the global configuration.
func WithContext(c *core.Context) Option { return func(t *Tasks) { t.Context = c } }

// Tasks runs the operations behind the workers; call its methods directly
// to run a task in-process (pyllym's task_always_eager).
type Tasks struct {
	Context *core.Context
}

// Register adds a worker for every kind to workers.
func Register(workers *river.Workers, opts ...Option) (*Tasks, error) {
	t := &Tasks{}
	for _, o := range opts {
		o(t)
	}
	for _, err := range []error{
		river.AddWorkerSafely(workers, &AskWorker{Tasks: t}),
		river.AddWorkerSafely(workers, &EmbedWorker{Tasks: t}),
		river.AddWorkerSafely(workers, &PaintWorker{Tasks: t}),
		river.AddWorkerSafely(workers, &SpeakWorker{Tasks: t}),
		river.AddWorkerSafely(workers, &TranscribeWorker{Tasks: t}),
		river.AddWorkerSafely(workers, &ModerateWorker{Tasks: t}),
	} {
		if err != nil {
			return nil, err
		}
	}
	return t, nil
}

func (t *Tasks) config() *core.Config {
	if t.Context != nil {
		return t.Context.Config
	}
	return core.Global()
}

func (t *Tasks) modelOptions(model, provider string) core.ModelOptions {
	return core.ModelOptions{Model: model, Provider: provider, Context: t.Context}
}

// --- ask ----------------------------------------------------------------------

// ThinkingArgs are with_thinking's keyword arguments.
type ThinkingArgs struct {
	Effort string `json:"effort,omitempty"`
	Budget *int   `json:"budget,omitempty"`
}

// AskArgs are the ask task's arguments. Prompt is the user message (a
// string, or nil with only attachments); Messages are prior messages in
// to_dict form; With lists attachment paths or URLs.
type AskArgs struct {
	Prompt       any              `json:"prompt,omitempty"`
	Model        string           `json:"model,omitempty"`
	Provider     string           `json:"provider,omitempty"`
	Instructions *string          `json:"instructions,omitempty"`
	Temperature  *float64         `json:"temperature,omitempty"`
	Schema       map[string]any   `json:"schema,omitempty"`
	Thinking     *ThinkingArgs    `json:"thinking,omitempty"`
	Messages     []map[string]any `json:"messages,omitempty"`
	With         any              `json:"with,omitempty"`
	Params       map[string]any   `json:"params,omitempty"`
	// AssumeModelExists skips the registry lookup (needs Provider); a
	// gollym addition for models missing from the registry.
	AssumeModelExists bool `json:"assume_model_exists,omitempty"`
}

func (AskArgs) Kind() string { return KindAsk }

// Ask runs the ask task and returns the assistant message's to_dict form.
func (t *Tasks) Ask(ctx context.Context, a AskArgs) (map[string]any, error) {
	chat, err := core.NewChat(core.ChatOptions{Model: a.Model, Provider: a.Provider,
		AssumeModelExists: a.AssumeModelExists, Context: t.Context})
	if err != nil {
		return nil, err
	}
	if a.Instructions != nil {
		chat.WithInstructions(*a.Instructions)
	}
	if a.Temperature != nil {
		chat.WithTemperature(*a.Temperature)
	}
	if a.Schema != nil {
		if _, err := chat.WithSchema(a.Schema); err != nil {
			return nil, err
		}
	}
	if a.Thinking != nil {
		if _, err := chat.WithThinking(a.Thinking.Effort, a.Thinking.Budget); err != nil {
			return nil, err
		}
	}
	if len(a.Params) > 0 {
		chat.WithParams(a.Params)
	}
	for _, m := range a.Messages {
		msg, err := MessageFromMap(m)
		if err != nil {
			return nil, err
		}
		if _, err := chat.AddMessage(msg); err != nil {
			return nil, err
		}
	}
	var opts []core.AskOption
	if a.With != nil {
		opts = append(opts, core.With(a.With))
	}
	reply, err := chat.Ask(ctx, a.Prompt, opts...)
	if err != nil {
		return nil, err
	}
	return MessageMap(reply), nil
}

// AskWorker works ask jobs.
type AskWorker struct {
	river.WorkerDefaults[AskArgs]
	Tasks *Tasks
}

func (w *AskWorker) Work(ctx context.Context, job *river.Job[AskArgs]) error {
	return record(ctx)(w.Tasks.Ask(ctx, job.Args))
}

// --- embed --------------------------------------------------------------------

// EmbedArgs are the embed task's arguments; Text is a string or a list of
// strings.
type EmbedArgs struct {
	Text       any    `json:"text"`
	Model      string `json:"model,omitempty"`
	Provider   string `json:"provider,omitempty"`
	Dimensions *int   `json:"dimensions,omitempty"`
}

func (EmbedArgs) Kind() string { return KindEmbed }

// Embed runs the embed task: {vectors, model, input_tokens}.
func (t *Tasks) Embed(ctx context.Context, a EmbedArgs) (map[string]any, error) {
	text := a.Text
	if list, ok := text.([]any); ok { // JSON lists arrive as []any
		strs := make([]string, len(list))
		for i, v := range list {
			strs[i] = fmt.Sprint(v)
		}
		text = strs
	}
	res, err := core.Embed(ctx, text, core.EmbedOptions{ModelOptions: t.modelOptions(a.Model, a.Provider), Dimensions: a.Dimensions})
	if err != nil {
		return nil, err
	}
	return map[string]any{"vectors": res.Vectors, "model": res.Model, "input_tokens": res.InputTokens}, nil
}

// EmbedWorker works embed jobs.
type EmbedWorker struct {
	river.WorkerDefaults[EmbedArgs]
	Tasks *Tasks
}

func (w *EmbedWorker) Work(ctx context.Context, job *river.Job[EmbedArgs]) error {
	return record(ctx)(w.Tasks.Embed(ctx, job.Args))
}

// --- paint --------------------------------------------------------------------

// PaintArgs are the paint task's arguments. With SavePath the image is
// written there and its data left out of the result.
type PaintArgs struct {
	Prompt   string         `json:"prompt"`
	Model    string         `json:"model,omitempty"`
	Provider string         `json:"provider,omitempty"`
	Size     string         `json:"size,omitempty"` // default 1024x1024
	SavePath string         `json:"save_path,omitempty"`
	Params   map[string]any `json:"params,omitempty"`
}

func (PaintArgs) Kind() string { return KindPaint }

// Paint runs the paint task: {url, data, mime_type, revised_prompt,
// model_id, path} without empty entries.
func (t *Tasks) Paint(ctx context.Context, a PaintArgs) (map[string]any, error) {
	img, err := core.Paint(ctx, a.Prompt, core.PaintRequest{ModelOptions: t.modelOptions(a.Model, a.Provider), Size: a.Size, Params: a.Params})
	if err != nil {
		return nil, err
	}
	data := img.Data
	if a.SavePath != "" {
		if err := img.Save(ctx, a.SavePath); err != nil {
			return nil, err
		}
		data = ""
	}
	return compact(map[string]any{
		"url": img.URL, "data": data, "mime_type": img.MIMEType, "revised_prompt": img.RevisedPrompt,
		"model_id": img.ModelID, "path": a.SavePath,
	}), nil
}

// PaintWorker works paint jobs.
type PaintWorker struct {
	river.WorkerDefaults[PaintArgs]
	Tasks *Tasks
}

func (w *PaintWorker) Work(ctx context.Context, job *river.Job[PaintArgs]) error {
	return record(ctx)(w.Tasks.Paint(ctx, job.Args))
}

// --- speak --------------------------------------------------------------------

// SpeakArgs are the speak task's arguments.
type SpeakArgs struct {
	Input    string `json:"input"`
	Model    string `json:"model,omitempty"`
	Provider string `json:"provider,omitempty"`
	Voice    string `json:"voice,omitempty"`
	Format   string `json:"format,omitempty"`
	SavePath string `json:"save_path,omitempty"`
}

func (SpeakArgs) Kind() string { return KindSpeak }

// Speak runs the speak task: {data_base64, model, voice, format,
// mime_type, path} without empty entries.
func (t *Tasks) Speak(ctx context.Context, a SpeakArgs) (map[string]any, error) {
	speech, err := core.Speak(ctx, a.Input, core.SpeakRequest{ModelOptions: t.modelOptions(a.Model, a.Provider), Voice: a.Voice, Format: a.Format})
	if err != nil {
		return nil, err
	}
	data := ""
	if a.SavePath != "" {
		if err := speech.Save(a.SavePath); err != nil {
			return nil, err
		}
	} else {
		data = base64.StdEncoding.EncodeToString(speech.Data)
	}
	return compact(map[string]any{
		"data_base64": data, "model": speech.Model, "voice": speech.Voice, "format": speech.Format,
		"mime_type": speech.MIMEType, "path": a.SavePath,
	}), nil
}

// SpeakWorker works speak jobs.
type SpeakWorker struct {
	river.WorkerDefaults[SpeakArgs]
	Tasks *Tasks
}

func (w *SpeakWorker) Work(ctx context.Context, job *river.Job[SpeakArgs]) error {
	return record(ctx)(w.Tasks.Speak(ctx, job.Args))
}

// --- transcribe ---------------------------------------------------------------

// TranscribeArgs are the transcribe task's arguments.
type TranscribeArgs struct {
	AudioFile string `json:"audio_file"`
	Model     string `json:"model,omitempty"`
	Provider  string `json:"provider,omitempty"`
	Language  string `json:"language,omitempty"`
}

func (TranscribeArgs) Kind() string { return KindTranscribe }

// Transcribe runs the transcribe task: {text, model, language, duration,
// segments, words, input_tokens, output_tokens} without empty entries.
func (t *Tasks) Transcribe(ctx context.Context, a TranscribeArgs) (map[string]any, error) {
	res, err := core.Transcribe(ctx, a.AudioFile, core.TranscribeRequest{ModelOptions: t.modelOptions(a.Model, a.Provider), Language: a.Language})
	if err != nil {
		return nil, err
	}
	out := compact(map[string]any{
		"model": res.Model, "language": res.Language, "duration": res.Duration, "segments": res.Segments,
		"words": res.Words, "input_tokens": res.InputTokens, "output_tokens": res.OutputTokens,
	})
	out["text"] = res.Text // always a string in pyllym
	return out, nil
}

// TranscribeWorker works transcribe jobs.
type TranscribeWorker struct {
	river.WorkerDefaults[TranscribeArgs]
	Tasks *Tasks
}

func (w *TranscribeWorker) Work(ctx context.Context, job *river.Job[TranscribeArgs]) error {
	return record(ctx)(w.Tasks.Transcribe(ctx, job.Args))
}

// --- moderate -----------------------------------------------------------------

// ModerateArgs are the moderate task's arguments; Input is a string or a
// list.
type ModerateArgs struct {
	Input    any    `json:"input"`
	Model    string `json:"model,omitempty"`
	Provider string `json:"provider,omitempty"`
}

func (ModerateArgs) Kind() string { return KindModerate }

// Moderate runs the moderate task: {id, model, results, flagged}.
func (t *Tasks) Moderate(ctx context.Context, a ModerateArgs) (map[string]any, error) {
	res, err := core.Moderate(ctx, a.Input, t.modelOptions(a.Model, a.Provider))
	if err != nil {
		return nil, err
	}
	results := res.Results
	if results == nil {
		results = []map[string]any{}
	}
	return map[string]any{"id": res.ID, "model": res.Model, "results": results, "flagged": res.IsFlagged()}, nil
}

// ModerateWorker works moderate jobs.
type ModerateWorker struct {
	river.WorkerDefaults[ModerateArgs]
	Tasks *Tasks
}

func (w *ModerateWorker) Work(ctx context.Context, job *river.Job[ModerateArgs]) error {
	return record(ctx)(w.Tasks.Moderate(ctx, job.Args))
}

// --- helpers ------------------------------------------------------------------

// record stores a task's result as the job output.
func record(ctx context.Context) func(map[string]any, error) error {
	return func(out map[string]any, err error) error {
		if err != nil {
			return err
		}
		return river.RecordOutput(ctx, out)
	}
}

// compact drops nil, empty-string and nil-pointer entries (pyllym drops
// None; gollym uses "" for unset strings).
func compact(m map[string]any) map[string]any {
	out := make(map[string]any, len(m))
	for k, v := range m {
		switch x := v.(type) {
		case nil:
			continue
		case string:
			if x == "" {
				continue
			}
		case *float64:
			if x == nil {
				continue
			}
			out[k] = *x
			continue
		case *int:
			if x == nil {
				continue
			}
			out[k] = *x
			continue
		}
		out[k] = v
	}
	return out
}
