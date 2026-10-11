package core

import (
	"errors"
	"fmt"
	"strings"
	"time"
)

// ErrorKind classifies provider errors (pyllym's Error subclasses).
type ErrorKind string

const (
	KindError                 ErrorKind = "error"
	KindConnectionFailed      ErrorKind = "connection_failed"
	KindBadRequest            ErrorKind = "bad_request"
	KindUnauthorized          ErrorKind = "unauthorized"
	KindPaymentRequired       ErrorKind = "payment_required"
	KindForbidden             ErrorKind = "forbidden"
	KindContextLengthExceeded ErrorKind = "context_length_exceeded"
	KindRateLimit             ErrorKind = "rate_limit"
	KindServer                ErrorKind = "server"
	KindServiceUnavailable    ErrorKind = "service_unavailable"
	KindOverloaded            ErrorKind = "overloaded"
)

var defaultMessages = map[ErrorKind]string{
	KindConnectionFailed:      "Connection failed - unable to reach the provider",
	KindBadRequest:            "Invalid request - please check your input",
	KindForbidden:             "Forbidden - you do not have permission to access this resource",
	KindContextLengthExceeded: "Context length exceeded",
	KindOverloaded:            "Service overloaded - please try again later",
	KindPaymentRequired:       "Payment required - please top up your account",
	KindRateLimit:             "Rate limit exceeded - please wait a moment",
	KindServer:                "API server error - please try again",
	KindServiceUnavailable:    "API server unavailable - please try again later",
	KindUnauthorized:          "Invalid API key - check your credentials",
}

// Error is a provider/API error. Match kinds with errors.Is against the
// Err* sentinels: errors.Is(err, core.ErrRateLimit).
type Error struct {
	Kind     ErrorKind
	Message  string
	Response *Response // nil for transport failures
	// RetryAfter is how long the provider asked us to wait, when it said.
	RetryAfter *time.Duration
	Cause      error
}

func (e *Error) Error() string { return e.Message }
func (e *Error) Unwrap() error { return e.Cause }

// Is matches an *Error against a sentinel of the same kind. KindError is the
// base: every *Error matches ErrAPI.
func (e *Error) Is(target error) bool {
	t, ok := target.(*Error)
	if !ok || t.Message != sentinelMessage {
		return false
	}
	return t.Kind == KindError || t.Kind == e.Kind
}

// Status is the HTTP status of the failed response, or 0.
func (e *Error) Status() int {
	if e.Response == nil {
		return 0
	}
	return e.Response.Status
}

const sentinelMessage = "\x00sentinel"

func sentinel(k ErrorKind) *Error { return &Error{Kind: k, Message: sentinelMessage} }

// Sentinels for errors.Is.
var (
	ErrAPI                   = sentinel(KindError)
	ErrConnectionFailed      = sentinel(KindConnectionFailed)
	ErrBadRequest            = sentinel(KindBadRequest)
	ErrUnauthorized          = sentinel(KindUnauthorized)
	ErrPaymentRequired       = sentinel(KindPaymentRequired)
	ErrForbidden             = sentinel(KindForbidden)
	ErrContextLengthExceeded = sentinel(KindContextLengthExceeded)
	ErrRateLimit             = sentinel(KindRateLimit)
	ErrServer                = sentinel(KindServer)
	ErrServiceUnavailable    = sentinel(KindServiceUnavailable)
	ErrOverloaded            = sentinel(KindOverloaded)
)

// NewError builds an *Error of kind k. An empty message falls back to the
// response body (when textual) and then the kind's default message, as
// pyllym's Error.__init__ does.
func NewError(k ErrorKind, resp *Response, message string) *Error {
	if message == "" && resp != nil {
		switch b := resp.Body.(type) {
		case string:
			message = b
		case []byte:
			message = string(b)
		default:
			if Truthy(b) {
				message = PyStr(b)
			}
		}
	}
	if message == "" {
		message = defaultMessages[k]
	}
	return &Error{Kind: k, Message: message, Response: resp}
}

// WrapConnectionError reports a transport failure (DNS, refused, timeout,
// dropped stream) as a ConnectionFailed error keeping the cause.
func WrapConnectionError(err error) *Error {
	return &Error{Kind: KindConnectionFailed, Message: err.Error(), Cause: err}
}

// Retryable reports whether the connection retries this kind.
func (k ErrorKind) Retryable() bool {
	switch k {
	case KindRateLimit, KindServer, KindServiceUnavailable, KindOverloaded:
		return true
	}
	return false
}

var statusKinds = map[int]ErrorKind{
	400: KindBadRequest,
	401: KindUnauthorized,
	402: KindPaymentRequired,
	403: KindForbidden,
	429: KindRateLimit,
	500: KindServer,
	502: KindServiceUnavailable,
	503: KindServiceUnavailable,
	504: KindServiceUnavailable,
	529: KindOverloaded,
}

var contextLengthPatterns = []string{
	"context length",
	"context_length_exceeded",
	"context window",
	"maximum context",
	"prompt is too long",
	"too many tokens",
	"input is too long",
	"exceeds the maximum number of tokens",
}

// IsContextLengthMessage reports a context/token-limit overflow message.
func IsContextLengthMessage(message string) bool {
	if message == "" {
		return false
	}
	lowered := strings.ToLower(message)
	for _, p := range contextLengthPatterns {
		if strings.Contains(lowered, p) {
			return true
		}
	}
	return false
}

// Exhausted balance/quota reported as 429 is a billing problem, not
// throttling: waiting can never help.
var quotaPatterns = []string{
	"insufficient_quota",
	"insufficient quota",
	"exceeded your current quota",
	"insufficient balance",
	"insufficient_balance",
	"please recharge",
	"余额不足",
	"欠费",
}

// IsQuotaMessage reports an exhausted-balance message.
func IsQuotaMessage(message string) bool {
	if message == "" {
		return false
	}
	lowered := strings.ToLower(message)
	for _, p := range quotaPatterns {
		if strings.Contains(lowered, p) {
			return true
		}
	}
	return false
}

// ErrorKindForStatus is pyllym's error_for_status.
func ErrorKindForStatus(status int, message string) ErrorKind {
	if (status == 400 || status == 413 || status == 429) && IsContextLengthMessage(message) {
		return KindContextLengthExceeded
	}
	if status == 429 && IsQuotaMessage(message) {
		return KindPaymentRequired
	}
	if k, ok := statusKinds[status]; ok {
		return k
	}
	if status >= 400 && status < 500 {
		return KindBadRequest
	}
	return KindServer
}

// Non-HTTP errors.
var (
	ErrConfiguration      = errors.New("configuration error")
	ErrModelNotFound      = errors.New("model not found")
	ErrInvalidRole        = errors.New("invalid role")
	ErrInvalidToolChoice  = errors.New("invalid tool choice")
	ErrUnsupported        = errors.New("not supported")
	ErrUnsupportedContent = errors.New("unsupported attachment")
)

// UnsupportedAttachmentError is raised when an attachment can't be formatted
// for the selected provider.
type UnsupportedAttachmentError struct{ Type string }

func (e *UnsupportedAttachmentError) Error() string {
	msg := "Unsupported attachment type"
	if e.Type != "" {
		msg += ": " + e.Type
	}
	return msg + ". Consider using a model that supports this attachment type."
}

func (e *UnsupportedAttachmentError) Is(target error) bool { return target == ErrUnsupportedContent }

// ConfigurationError is raised when a provider is used without its required
// options.
type ConfigurationError struct{ Message string }

func (e *ConfigurationError) Error() string        { return e.Message }
func (e *ConfigurationError) Is(target error) bool { return target == ErrConfiguration }

// ModelNotFoundError is raised by registry lookups.
type ModelNotFoundError struct{ Message string }

func (e *ModelNotFoundError) Error() string        { return e.Message }
func (e *ModelNotFoundError) Is(target error) bool { return target == ErrModelNotFound }

// NotSupported reports a feature a protocol doesn't implement.
func NotSupported(what string) error { return fmt.Errorf("%w: %s", ErrUnsupported, what) }
