package core

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

// Embedding is the result of Embed. Vectors is []float64 for one input
// string, [][]float64 for a list input.
type Embedding struct {
	Vectors     any
	Model       string
	InputTokens int
}

// Moderation is the result of Moderate.
type Moderation struct {
	ID      string
	Model   string
	Results []map[string]any
}

// IsFlagged reports any flagged result.
func (m *Moderation) IsFlagged() bool {
	for _, r := range m.Results {
		if Truthy(r["flagged"]) {
			return true
		}
	}
	return false
}

// FlaggedCategories lists flagged category names, first occurrence order.
func (m *Moderation) FlaggedCategories() []string {
	seen := map[string]bool{}
	var out []string
	for _, r := range m.Results {
		cats, _ := r["categories"].(map[string]any)
		for _, name := range sortedKeys(cats) {
			if Truthy(cats[name]) && !seen[name] {
				seen[name] = true
				out = append(out, name)
			}
		}
	}
	return out
}

// CategoryScores of the first result.
func (m *Moderation) CategoryScores() map[string]any {
	if len(m.Results) == 0 {
		return map[string]any{}
	}
	s, _ := m.Results[0]["category_scores"].(map[string]any)
	if s == nil {
		return map[string]any{}
	}
	return s
}

// Categories of the first result.
func (m *Moderation) Categories() map[string]any {
	if len(m.Results) == 0 {
		return map[string]any{}
	}
	s, _ := m.Results[0]["categories"].(map[string]any)
	if s == nil {
		return map[string]any{}
	}
	return s
}

// Image is a generated image (a URL or base64 data).
type Image struct {
	URL           string
	Data          string // base64
	MIMEType      string
	RevisedPrompt string
	ModelID       string
	Usage         map[string]any
}

func (i *Image) IsBase64() bool { return i.Data != "" }

// Blob returns the image bytes, fetching URLs (60 s timeout).
func (i *Image) Blob(ctx context.Context) ([]byte, error) {
	if i.IsBase64() {
		return base64.StdEncoding.DecodeString(i.Data)
	}
	return fetchURL(ctx, i.URL, 60*time.Second)
}

// Save writes the image to path.
func (i *Image) Save(ctx context.Context, path string) error {
	b, err := i.Blob(ctx)
	if err != nil {
		return err
	}
	return os.WriteFile(expandHome(path), b, 0o644)
}

// Video is a generated video.
type Video struct {
	URL      string
	Data     string // base64
	MIMEType string
	ModelID  string
	Duration *float64
	Raw      any
}

func (v *Video) IsBase64() bool { return v.Data != "" }

// Blob returns the video bytes, fetching URLs (300 s timeout).
func (v *Video) Blob(ctx context.Context) ([]byte, error) {
	if v.IsBase64() {
		return base64.StdEncoding.DecodeString(v.Data)
	}
	return fetchURL(ctx, v.URL, 300*time.Second)
}

// Save writes the video to path.
func (v *Video) Save(ctx context.Context, path string) error {
	b, err := v.Blob(ctx)
	if err != nil {
		return err
	}
	return os.WriteFile(expandHome(path), b, 0o644)
}

var speechMIME = map[string]string{
	"aac": "audio/aac", "flac": "audio/flac", "mp3": "audio/mpeg",
	"opus": "audio/opus", "pcm": "audio/pcm", "wav": "audio/wav",
}

// Speech is synthesised audio.
type Speech struct {
	Data     []byte
	Model    string
	Voice    string
	Format   string
	MIMEType string
}

// NewSpeech fills Format ("mp3") and MIMEType defaults.
func NewSpeech(data []byte, model, voice, format, mimeType string) *Speech {
	if format == "" {
		format = "mp3"
	}
	if mimeType == "" {
		mimeType = speechMIME[format]
		if mimeType == "" {
			mimeType = "audio/" + format
		}
	}
	return &Speech{Data: data, Model: model, Voice: voice, Format: format, MIMEType: mimeType}
}

// Save writes the audio to path.
func (s *Speech) Save(path string) error { return os.WriteFile(expandHome(path), s.Data, 0o644) }

// Transcription is speech-to-text output.
type Transcription struct {
	Text         string
	Model        string
	Language     string
	Duration     *float64
	Segments     any
	Words        any
	InputTokens  *int
	OutputTokens *int
}

// LayoutElement is one recognised block on a page.
type LayoutElement struct {
	Index   any
	Label   any
	BBox    any
	Content any
	Height  any
	Width   any
}

// NewLayoutElement reads a wire element.
func NewLayoutElement(d map[string]any) LayoutElement {
	return LayoutElement{Index: d["index"], Label: d["label"], BBox: d["bbox_2d"], Content: d["content"], Height: d["height"], Width: d["width"]}
}

// Map is the to_dict form.
func (e LayoutElement) Map() map[string]any {
	return map[string]any{"index": e.Index, "label": e.Label, "bbox_2d": e.BBox, "content": e.Content, "height": e.Height, "width": e.Width}
}

// LayoutParsing is document OCR output: Markdown text plus layout elements.
type LayoutParsing struct {
	Text          string
	Model         string
	ID            string
	Created       any
	Elements      []LayoutElement
	Visualization any
	PageCount     *int
	Pages         any
	InputTokens   *int
	OutputTokens  *int
	CachedTokens  *int
}

// Markdown is the parsed text.
func (l *LayoutParsing) Markdown() string { return l.Text }

// UploadedFile is a provider-managed file.
type UploadedFile struct {
	ID           string
	Provider     string
	Filename     string
	ByteSize     *int64
	CreatedAt    *time.Time
	ExpiresAt    *time.Time
	Status       string
	MIMEType     string
	Purpose      string
	URI          string
	Downloadable *bool
	Metadata     map[string]any
}

func fetchURL(ctx context.Context, url string, timeout time.Duration) ([]byte, error) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, WrapConnectionError(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return nil, NewError(ErrorKindForStatus(resp.StatusCode, ""), nil, fmt.Sprintf("HTTP %d fetching %s", resp.StatusCode, url))
	}
	return io.ReadAll(resp.Body)
}
