package core

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

var documentExtensions = map[string]bool{
	"doc": true, "docx": true, "dot": true, "key": true, "numbers": true, "odp": true,
	"ods": true, "odt": true, "pages": true, "pot": true, "pps": true, "ppt": true,
	"pptx": true, "rtf": true, "xls": true, "xlsx": true,
}

// Attachment is a file sent with a message. Exactly one source is set: a
// URL, a filesystem path, in-memory bytes, a reader, or a provider-managed
// UploadedFile.
type Attachment struct {
	URL          string
	Path         string
	Data         []byte
	Reader       io.Reader
	ProviderFile *UploadedFile

	Filename string
	MIMEType string

	once    sync.Once
	content []byte
	loadErr error
}

// NewAttachment classifies a source like pyllym's Attachment: http(s)
// strings are URLs, other strings are paths; []byte and io.Reader are
// in-memory; *UploadedFile is a provider file; *Attachment is returned as is
// (with filename overridden when given).
func NewAttachment(source any, filename string) (*Attachment, error) {
	a := &Attachment{}
	switch s := source.(type) {
	case *Attachment:
		if filename == "" {
			return s, nil
		}
		cp := &Attachment{URL: s.URL, Path: s.Path, Data: s.Data, Reader: s.Reader, ProviderFile: s.ProviderFile}
		a = cp
	case string:
		if strings.HasPrefix(s, "http://") || strings.HasPrefix(s, "https://") {
			a.URL = s
		} else {
			a.Path = s
		}
	case []byte:
		a.Data = s
	case *UploadedFile:
		a.ProviderFile = s
	case io.Reader:
		a.Reader = s
	default:
		return nil, fmt.Errorf("unsupported attachment source %T", source)
	}
	a.Filename = filename
	if a.Filename == "" {
		a.Filename = a.sourceFilename()
	}
	a.MIMEType = a.determineMIME()
	return a, nil
}

// MustAttachment is NewAttachment that panics on an unsupported source type.
func MustAttachment(source any, filename string) *Attachment {
	a, err := NewAttachment(source, filename)
	if err != nil {
		panic(err)
	}
	return a
}

func (a *Attachment) IsURL() bool          { return a.URL != "" }
func (a *Attachment) IsPath() bool         { return a.Path != "" }
func (a *Attachment) IsProviderFile() bool { return a.ProviderFile != nil }
func (a *Attachment) IsIOLike() bool       { return a.Reader != nil }

// ProviderFileID is the id of a provider-managed file ("" otherwise).
func (a *Attachment) ProviderFileID() string {
	if a.ProviderFile == nil {
		return ""
	}
	return a.ProviderFile.ID
}

// ProviderFileURI is the URI of a provider-managed file ("" otherwise).
func (a *Attachment) ProviderFileURI() string {
	if a.ProviderFile == nil {
		return ""
	}
	return a.ProviderFile.URI
}

// Source returns the original source value.
func (a *Attachment) Source() any {
	switch {
	case a.URL != "":
		return a.URL
	case a.Path != "":
		return a.Path
	case a.ProviderFile != nil:
		return a.ProviderFile
	case a.Reader != nil:
		return a.Reader
	}
	return a.Data
}

// Content loads the bytes (once). URLs are fetched with a 30 s timeout.
func (a *Attachment) Content() ([]byte, error) {
	if a.ProviderFile != nil {
		return nil, fmt.Errorf("Provider-managed file %s cannot be read as inline attachment content", a.ProviderFile.ID)
	}
	a.once.Do(func() { a.content, a.loadErr = a.load() })
	return a.content, a.loadErr
}

func (a *Attachment) load() ([]byte, error) {
	switch {
	case a.URL != "":
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, a.URL, nil)
		if err != nil {
			return nil, err
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return nil, WrapConnectionError(err)
		}
		defer resp.Body.Close()
		if resp.StatusCode >= 400 {
			return nil, NewError(ErrorKindForStatus(resp.StatusCode, ""), nil, fmt.Sprintf("HTTP %d fetching %s", resp.StatusCode, a.URL))
		}
		return io.ReadAll(resp.Body)
	case a.Path != "":
		return os.ReadFile(expandHome(a.Path))
	case a.Data != nil:
		return a.Data, nil
	case a.Reader != nil:
		if s, ok := a.Reader.(io.Seeker); ok {
			_, _ = s.Seek(0, io.SeekStart)
		}
		return io.ReadAll(a.Reader)
	}
	return nil, nil
}

// Encoded is the base64 of the content.
func (a *Attachment) Encoded() (string, error) {
	data, err := a.Content()
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(data), nil
}

// ForLLM renders text inline as <file ...> and everything else as a data URL.
func (a *Attachment) ForLLM() (string, error) {
	if a.Type() == "text" {
		data, err := a.Content()
		if err != nil {
			return "", err
		}
		return fmt.Sprintf("<file name='%s' mime_type='%s'>%s</file>", a.Filename, a.MIMEType, strings.ToValidUTF8(string(data), "�")), nil
	}
	enc, err := a.Encoded()
	if err != nil {
		return "", err
	}
	return "data:" + a.MIMEType + ";base64," + enc, nil
}

// Type is image, video, audio, pdf, text, document or unknown.
func (a *Attachment) Type() string {
	switch {
	case a.IsImage():
		return "image"
	case a.IsVideo():
		return "video"
	case a.IsAudio():
		return "audio"
	case a.IsPDF():
		return "pdf"
	case a.IsText():
		return "text"
	case a.IsDocument():
		return "document"
	}
	return "unknown"
}

func (a *Attachment) IsImage() bool { return IsImageMIME(a.MIMEType) }
func (a *Attachment) IsVideo() bool { return IsVideoMIME(a.MIMEType) }
func (a *Attachment) IsAudio() bool { return IsAudioMIME(a.MIMEType) }
func (a *Attachment) IsPDF() bool   { return IsPDFMIME(a.MIMEType) }
func (a *Attachment) IsText() bool  { return IsTextMIME(a.MIMEType) }

func (a *Attachment) IsDocument() bool {
	if a.IsPDF() || a.IsText() {
		return false
	}
	return IsDocumentMIME(a.MIMEType) || documentExtensions[a.Extension()]
}

// Format is the short format name (mp3, wav, or the MIME subtype).
func (a *Attachment) Format() string {
	switch a.MIMEType {
	case "audio/mpeg":
		return "mp3"
	case "audio/wav", "audio/wave", "audio/x-wav":
		return "wav"
	}
	parts := strings.Split(a.MIMEType, "/")
	return parts[len(parts)-1]
}

// Extension is the lowercase filename extension without the dot.
func (a *Attachment) Extension() string {
	return strings.ToLower(strings.TrimPrefix(pyExt(a.Filename), "."))
}

// ByteSize returns the size when it is known without reading a URL.
func (a *Attachment) ByteSize() (int64, bool) {
	switch {
	case a.ProviderFile != nil:
		if a.ProviderFile.ByteSize != nil {
			return *a.ProviderFile.ByteSize, true
		}
		return 0, false
	case a.Path != "":
		st, err := os.Stat(expandHome(a.Path))
		if err != nil {
			return 0, false
		}
		return st.Size(), true
	case a.Data != nil:
		return int64(len(a.Data)), true
	}
	if a.content != nil {
		return int64(len(a.content)), true
	}
	return 0, false
}

// Map is the to_dict form.
func (a *Attachment) Map() map[string]any {
	return map[string]any{"type": a.Type(), "source": a.Source()}
}

func (a *Attachment) determineMIME() string {
	if a.ProviderFile != nil {
		if a.ProviderFile.MIMEType != "" {
			return a.ProviderFile.MIMEType
		}
		return MIMEForSource(nil, a.ProviderFile.Filename)
	}
	t := MIMEForName(a.Filename)
	if t == "" && a.Path != "" {
		t = MIMEForName(a.Path)
	}
	if t == "" && a.URL == "" {
		if data, err := a.Content(); err == nil && len(data) > 0 {
			t = SniffMIME(data)
		}
	}
	if t == "" {
		t = DefaultMIME
	}
	if t == "audio/x-wav" {
		t = "audio/wav"
	}
	return t
}

func (a *Attachment) sourceFilename() string {
	switch {
	case a.URL != "":
		// os.path.basename: the part after the last slash ("" for "/dir/").
		if u, err := url.Parse(a.URL); err == nil {
			if base := u.Path[strings.LastIndex(u.Path, "/")+1:]; base != "" {
				return base
			}
		}
		return "attachment"
	case a.ProviderFile != nil:
		if a.ProviderFile.Filename != "" {
			return a.ProviderFile.Filename
		}
		return "attachment"
	case a.Path != "":
		return filepath.Base(a.Path)
	case a.Reader != nil:
		if n, ok := a.Reader.(interface{ Name() string }); ok && n.Name() != "" {
			return filepath.Base(n.Name())
		}
		return "attachment"
	}
	return ""
}

func expandHome(p string) string {
	if strings.HasPrefix(p, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, p[2:])
		}
	}
	return p
}
