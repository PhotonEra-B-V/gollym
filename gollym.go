// Package gollym is one interface for every major AI provider: a Go port of
// pyllym. Chat, stream, call tools, get structured output, embed, generate
// images and speech, transcribe, moderate and decide, through OpenAI,
// Anthropic, Gemini, Bedrock, Zhipu GLM, vLLM, Ollama and any
// OpenAI-compatible API.
//
//	gollym.Configure(func(c *gollym.Config) { c.Set("openai_api_key", key) })
//	chat, err := gollym.NewChat(gollym.ChatOptions{Model: "gpt-5.4"})
//	msg, err := chat.Ask(ctx, "What's the best way to learn Go?")
//	fmt.Println(msg.Text())
package gollym

import (
	"github.com/PhotonEra-B-V/gollym/core"
	_ "github.com/PhotonEra-B-V/gollym/providers" // register built-in providers
)

// Version is the pyllym release this port follows.
const Version = "1.16.0"

type (
	Config            = core.Config
	Context           = core.Context
	Chat              = core.Chat
	ChatOptions       = core.ChatOptions
	AskOption         = core.AskOption
	Message           = core.Message
	Chunk             = core.Chunk
	Content           = core.Content
	RawContent        = core.RawContent
	SearchResults     = core.SearchResults
	Attachment        = core.Attachment
	Tokens            = core.Tokens
	Thinking          = core.Thinking
	Citation          = core.Citation
	ToolCall          = core.ToolCall
	Tool              = core.Tool
	FuncTool          = core.FuncTool
	Parameter         = core.Parameter
	Info              = core.Info
	Cost              = core.Cost
	CostAggregate     = core.CostAggregate
	Error             = core.Error
	Embedding         = core.Embedding
	EmbedOptions      = core.EmbedOptions
	Moderation        = core.Moderation
	Image             = core.Image
	PaintRequest      = core.PaintRequest
	Video             = core.Video
	AnimateRequest    = core.AnimateRequest
	Speech            = core.Speech
	SpeakRequest      = core.SpeakRequest
	Transcription     = core.Transcription
	TranscribeRequest = core.TranscribeRequest
	LayoutParsing     = core.LayoutParsing
	LayoutRequest     = core.LayoutRequest
	Question          = core.Question
	Questions         = core.Questions
	Answer            = core.Answer
	Decision          = core.Decision
	DecideRequest     = core.DecideRequest
	ModelOptions      = core.ModelOptions
	UploadedFile      = core.UploadedFile
	FileOptions       = core.FileOptions
	Provider          = core.Provider
	ProviderSpec      = core.ProviderSpec
	Registry          = core.Registry
)

// Error sentinels for errors.Is.
var (
	ErrAPI                   = core.ErrAPI
	ErrConnectionFailed      = core.ErrConnectionFailed
	ErrBadRequest            = core.ErrBadRequest
	ErrUnauthorized          = core.ErrUnauthorized
	ErrPaymentRequired       = core.ErrPaymentRequired
	ErrForbidden             = core.ErrForbidden
	ErrContextLengthExceeded = core.ErrContextLengthExceeded
	ErrRateLimit             = core.ErrRateLimit
	ErrServer                = core.ErrServer
	ErrServiceUnavailable    = core.ErrServiceUnavailable
	ErrOverloaded            = core.ErrOverloaded
	ErrConfiguration         = core.ErrConfiguration
	ErrModelNotFound         = core.ErrModelNotFound
	ErrUnsupported           = core.ErrUnsupported
)

var (
	// Global returns the process-wide configuration.
	Global = core.Global
	// Configure applies fn to the global configuration.
	Configure = core.Configure
	// NewContext copies the global config for one scope.
	NewContext = core.NewContext
	// NewChat starts a conversation.
	NewChat = core.NewChat
	// With attaches files to an Ask.
	With = core.With
	// NewAttachment wraps a path, URL, bytes, reader or uploaded file.
	NewAttachment = core.NewAttachment
	// OnChunk streams an Ask.
	OnChunk = core.OnChunk
	// NewTool builds a function tool.
	NewTool = core.NewTool
	// Models is the model registry.
	Models = core.Models
	// Providers lists registered provider slugs.
	Providers = core.ProviderSlugs

	Embed       = core.Embed
	Moderate    = core.Moderate
	Paint       = core.Paint
	Animate     = core.Animate
	Speak       = core.Speak
	Transcribe  = core.Transcribe
	ParseLayout = core.ParseLayout
	Decide      = core.Decide
	Upload      = core.Upload
	FindFile    = core.FindFile
	Download    = core.Download
	ListModels  = core.ListModels
)
