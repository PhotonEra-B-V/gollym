package core

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"net/http"
	"path"
	"strings"
	"sync"
)

// Python's built-in mimetypes table (MimeTypes(filenames=()), strict), so
// extension lookups give the same answer as pyllym on every OS.
//
//go:embed mimetypes.json
var mimeTypesJSON []byte

var (
	mimeOnce  sync.Once
	mimeTable map[string]string
)

// DefaultMIME is the fallback type.
const DefaultMIME = "application/octet-stream"

var textSuffixes = []string{"+json", "+xml", "+html", "+yaml", "+csv", "+plain", "+javascript", "+svg"}

var nonTextPrefixTextMIME = map[string]bool{
	"application/json": true, "application/xml": true, "application/javascript": true,
	"application/ecmascript": true, "application/rtf": true, "application/sql": true,
	"application/x-sh": true, "application/x-csh": true, "application/x-httpd-php": true,
	"application/sdp": true, "application/sparql-query": true, "application/graphql": true,
	"application/yang": true, "application/mbox": true, "application/x-tex": true,
	"application/x-latex": true, "application/x-perl": true, "application/x-python": true,
	"application/x-tcl": true, "application/pgp-signature": true, "application/pgp-keys": true,
	"application/vnd.coffeescript": true, "application/vnd.dart": true,
	"application/vnd.oai.openapi": true, "application/vnd.zul": true, "application/x-yaml": true,
	"application/yaml": true, "application/toml": true,
}

var documentMIME = map[string]bool{
	"application/msword": true, "application/rtf": true, "application/vnd.apple.keynote": true,
	"application/vnd.apple.numbers": true, "application/vnd.apple.pages": true,
	"application/vnd.google-apps.document": true, "application/vnd.google-apps.presentation": true,
	"application/vnd.google-apps.spreadsheet": true, "application/vnd.ms-excel": true,
	"application/vnd.ms-powerpoint": true,
}

var documentMIMEPrefixes = []string{
	"application/vnd.openxmlformats-officedocument.",
	"application/vnd.oasis.opendocument.",
}

// MIMEForName guesses a type from a filename (Python mimetypes.guess_type).
func MIMEForName(name string) string {
	mimeOnce.Do(func() {
		mimeTable = map[string]string{}
		_ = json.Unmarshal(mimeTypesJSON, &mimeTable)
	})
	if name == "" {
		return ""
	}
	ext := strings.ToLower(path.Ext(name))
	if ext == "" {
		return ""
	}
	if t, ok := mimeTable[ext]; ok {
		return t
	}
	// guess_type is case-sensitive first, then lowercase.
	return mimeTable[path.Ext(name)]
}

// MIMEForSource is mime_type.for_source: name first, then a path source, then
// content sniffing.
func MIMEForSource(data []byte, name string) string {
	if t := MIMEForName(name); t != "" {
		return t
	}
	if len(data) > 0 {
		if t := SniffMIME(data); t != "" {
			return t
		}
	}
	return DefaultMIME
}

// SniffMIME detects common binary formats by magic bytes (pyllym's optional
// puremagic step), falling back to net/http's sniffer for the rest.
func SniffMIME(data []byte) string {
	switch {
	case bytes.HasPrefix(data, []byte("%PDF")):
		return "application/pdf"
	case bytes.HasPrefix(data, []byte("\x89PNG\r\n\x1a\n")):
		return "image/png"
	case bytes.HasPrefix(data, []byte("\xff\xd8\xff")):
		return "image/jpeg"
	case bytes.HasPrefix(data, []byte("GIF87a")), bytes.HasPrefix(data, []byte("GIF89a")):
		return "image/gif"
	case len(data) >= 12 && bytes.Equal(data[:4], []byte("RIFF")) && bytes.Equal(data[8:12], []byte("WEBP")):
		return "image/webp"
	case len(data) >= 12 && bytes.Equal(data[:4], []byte("RIFF")) && bytes.Equal(data[8:12], []byte("WAVE")):
		return "audio/wav"
	case bytes.HasPrefix(data, []byte("ID3")), len(data) > 1 && data[0] == 0xff && data[1]&0xe0 == 0xe0:
		return "audio/mpeg"
	case bytes.HasPrefix(data, []byte("fLaC")):
		return "audio/flac"
	case bytes.HasPrefix(data, []byte("OggS")):
		return "audio/ogg"
	case len(data) >= 12 && bytes.Equal(data[4:8], []byte("ftyp")):
		return "video/mp4"
	case bytes.HasPrefix(data, []byte("\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")):
		return "application/msword"
	case bytes.HasPrefix(data, []byte("{\\rtf")):
		return "application/rtf"
	}
	t := http.DetectContentType(data)
	if t == DefaultMIME {
		return ""
	}
	if i := strings.IndexByte(t, ';'); i >= 0 {
		t = t[:i]
	}
	return t
}

func IsImageMIME(t string) bool { return strings.HasPrefix(t, "image/") }
func IsVideoMIME(t string) bool { return strings.HasPrefix(t, "video/") }
func IsAudioMIME(t string) bool { return strings.HasPrefix(t, "audio/") }
func IsPDFMIME(t string) bool   { return t == "application/pdf" }

func IsTextMIME(t string) bool {
	if strings.HasPrefix(t, "text/") || nonTextPrefixTextMIME[t] {
		return true
	}
	for _, s := range textSuffixes {
		if strings.HasSuffix(t, s) {
			return true
		}
	}
	return false
}

func IsDocumentMIME(t string) bool {
	if IsPDFMIME(t) || IsTextMIME(t) {
		return false
	}
	if documentMIME[t] {
		return true
	}
	for _, p := range documentMIMEPrefixes {
		if strings.HasPrefix(t, p) {
			return true
		}
	}
	return false
}
