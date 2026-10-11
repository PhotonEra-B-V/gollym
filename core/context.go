package core

import (
	"context"
	"errors"
)

// Context is a per-call configuration scope: an isolated config copy.
type Context struct {
	Config *Config
}

// NewContext copies the global config and applies fn to the copy.
func NewContext(fn func(*Config)) *Context {
	cfg := Global().Copy()
	if fn != nil {
		fn(cfg)
	}
	return &Context{Config: cfg}
}

// NewChat creates a chat bound to this context.
func (c *Context) NewChat(opts ChatOptions) (*Chat, error) {
	opts.Context = c
	return NewChat(opts)
}

func cfgOf(c *Context) *Config {
	if c == nil {
		return Global()
	}
	return c.Config
}

// ModelOptions pick the model of a one-shot call.
type ModelOptions struct {
	Model             string
	Provider          string
	AssumeModelExists bool
	Context           *Context
}

func (o ModelOptions) resolve(def string) (*Info, Provider, error) {
	cfg := cfgOf(o.Context)
	model := o.Model
	if model == "" {
		model = def
	}
	return ResolveModel(model, o.Provider, o.AssumeModelExists, cfg)
}

// EmbedOptions configure Embed.
type EmbedOptions struct {
	ModelOptions
	Dimensions *int
}

// Embed embeds text (a string or a list of strings).
func Embed(ctx context.Context, text any, o EmbedOptions) (*Embedding, error) {
	info, p, err := o.resolve(cfgOf(o.Context).DefaultEmbeddingModel)
	if err != nil {
		return nil, err
	}
	f, err := AsFeature[Embedder](p, "embeddings")
	if err != nil {
		return nil, err
	}
	return f.Embed(ctx, text, info.ID, o.Dimensions)
}

// Moderate checks content against a moderation model.
func Moderate(ctx context.Context, input any, o ModelOptions) (*Moderation, error) {
	info, p, err := o.resolve(cfgOf(o.Context).DefaultModerationModel)
	if err != nil {
		return nil, err
	}
	f, err := AsFeature[Moderator](p, "moderation")
	if err != nil {
		return nil, err
	}
	return f.Moderate(ctx, input, info.ID)
}

// PaintRequest configures Paint.
type PaintRequest struct {
	ModelOptions
	Size   string // default 1024x1024
	With   any
	Mask   any
	Params map[string]any
}

// Paint generates an image.
func Paint(ctx context.Context, prompt string, o PaintRequest) (*Image, error) {
	info, p, err := o.resolve(cfgOf(o.Context).DefaultImageModel)
	if err != nil {
		return nil, err
	}
	f, err := AsFeature[Painter](p, "image generation")
	if err != nil {
		return nil, err
	}
	size := o.Size
	if size == "" {
		size = "1024x1024"
	}
	return f.Paint(ctx, prompt, PaintOptions{Model: info.ID, Size: size, With: o.With, Mask: o.Mask, Params: nonNilMap(o.Params)})
}

// AnimateRequest configures Animate (provider defaults to fal).
type AnimateRequest struct {
	ModelOptions
	Params map[string]any
}

// Animate generates a video.
func Animate(ctx context.Context, prompt string, o AnimateRequest) (*Video, error) {
	if o.Provider == "" {
		o.Provider = "fal"
	}
	def := cfgOf(o.Context).DefaultVideoModel
	if o.Model == "" && def == "" {
		return nil, errors.New("animate requires a model, e.g. model='fal-ai/ltx-video-13b-distilled'")
	}
	info, p, err := o.resolve(def)
	if err != nil {
		return nil, err
	}
	f, err := AsFeature[Animator](p, "video generation")
	if err != nil {
		return nil, err
	}
	return f.Animate(ctx, prompt, info.ID, nonNilMap(o.Params))
}

// SpeakRequest configures Speak.
type SpeakRequest struct {
	ModelOptions
	Voice        string
	Format       string
	Instructions string
	Speed        *float64
	Params       map[string]any
	Options      map[string]any
}

// Speak synthesises speech.
func Speak(ctx context.Context, input string, o SpeakRequest) (*Speech, error) {
	info, p, err := o.resolve(cfgOf(o.Context).DefaultSpeechModel)
	if err != nil {
		return nil, err
	}
	f, err := AsFeature[Speaker](p, "speech")
	if err != nil {
		return nil, err
	}
	return f.Speak(ctx, input, SpeakOptions{Model: info.ID, Voice: o.Voice, Format: o.Format, Instructions: o.Instructions,
		Speed: o.Speed, Params: nonNilMap(o.Params), Options: nonNilMap(o.Options)})
}

// TranscribeRequest configures Transcribe.
type TranscribeRequest struct {
	ModelOptions
	Language       string
	ResponseFormat string
	Prompt         string
	Temperature    *float64
	Options        map[string]any
}

// Transcribe converts an audio file to text.
func Transcribe(ctx context.Context, audioFile string, o TranscribeRequest) (*Transcription, error) {
	info, p, err := o.resolve(cfgOf(o.Context).DefaultTranscriptionModel)
	if err != nil {
		return nil, err
	}
	f, err := AsFeature[Transcriber](p, "transcription")
	if err != nil {
		return nil, err
	}
	return f.Transcribe(ctx, audioFile, TranscribeOptions{Model: info.ID, Language: o.Language, ResponseFormat: o.ResponseFormat,
		Prompt: o.Prompt, Temperature: o.Temperature, Options: nonNilMap(o.Options)})
}

// LayoutRequest configures ParseLayout.
type LayoutRequest struct {
	ModelOptions
	Options map[string]any
}

// ParseLayout runs document layout parsing (OCR).
func ParseLayout(ctx context.Context, file string, o LayoutRequest) (*LayoutParsing, error) {
	info, p, err := o.resolve(cfgOf(o.Context).DefaultLayoutParsingModel)
	if err != nil {
		return nil, err
	}
	f, err := AsFeature[LayoutParser](p, "layout parsing")
	if err != nil {
		return nil, err
	}
	return f.ParseLayout(ctx, file, info.ID, nonNilMap(o.Options))
}

// DecideRequest configures Decide.
type DecideRequest struct {
	ModelOptions
	Options map[string]any
}

// Decide answers typed questions about state with a decision model.
func Decide(ctx context.Context, state any, questions *Questions, o DecideRequest) (*Decision, error) {
	if questions == nil || len(questions.Names) == 0 {
		return nil, errors.New("decide() needs at least one question")
	}
	info, p, err := o.resolve(cfgOf(o.Context).DefaultDecisionModel)
	if err != nil {
		return nil, err
	}
	f, err := AsFeature[Decider](p, "decisions")
	if err != nil {
		return nil, err
	}
	return f.Decide(ctx, state, questions, info.ID, nonNilMap(o.Options))
}

// FileOptions pick the provider for file operations (default: the default
// model's provider).
type FileOptions struct {
	Provider string
	Context  *Context
}

func (o FileOptions) provider() (FileProvider, error) {
	cfg := cfgOf(o.Context)
	var p Provider
	if o.Provider != "" {
		spec, err := MustLookupProvider(o.Provider)
		if err != nil {
			return nil, err
		}
		if p, err = NewProvider(spec, cfg); err != nil {
			return nil, err
		}
	} else {
		_, prov, err := ResolveModel(cfg.DefaultModel, "", false, cfg)
		if err != nil {
			return nil, err
		}
		p = prov
	}
	fp, ok := p.(FileProvider)
	if !ok {
		return nil, NotSupported(p.Name() + " does not support file uploads")
	}
	return fp, nil
}

// Upload uploads a file to the provider.
func Upload(ctx context.Context, file any, o FileOptions, opts map[string]any) (*UploadedFile, error) {
	fp, err := o.provider()
	if err != nil {
		return nil, err
	}
	return fp.UploadFile(ctx, file, nonNilMap(opts))
}

// FindFile looks a provider file up.
func FindFile(ctx context.Context, id string, o FileOptions) (*UploadedFile, error) {
	fp, err := o.provider()
	if err != nil {
		return nil, err
	}
	return fp.FindFile(ctx, id)
}

// Download fetches a provider file's bytes.
func Download(ctx context.Context, id string, o FileOptions) ([]byte, error) {
	fp, err := o.provider()
	if err != nil {
		return nil, err
	}
	return fp.DownloadFile(ctx, id)
}

// ListModels asks a provider for its models.
func ListModels(ctx context.Context, p Provider) ([]*Info, error) {
	f, err := AsFeature[ModelLister](p, "listing models")
	if err != nil {
		return nil, err
	}
	return f.ListModels(ctx)
}

func nonNilMap(m map[string]any) map[string]any {
	if m == nil {
		return map[string]any{}
	}
	return m
}
