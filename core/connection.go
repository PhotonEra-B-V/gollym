package core

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"math/rand/v2"
	"mime/multipart"
	"net"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Response is the part of an HTTP response protocols use. Body is decoded
// JSON (map/slice) for JSON responses, a string for other text, and the raw
// bytes otherwise; Content always has the raw bytes.
type Response struct {
	Status  int
	Body    any
	Headers http.Header
	Content []byte
}

// BodyMap returns the body as a JSON object (nil otherwise).
func (r *Response) BodyMap() map[string]any {
	m, _ := r.Body.(map[string]any)
	return m
}

var textualTypes = []string{"application/json", "text/", "application/xml", "+json"}

// ParseBody decodes a response body like pyllym's _parse_body.
func ParseBody(contentType string, content []byte) any {
	textual := false
	for _, m := range textualTypes {
		if strings.Contains(contentType, m) {
			textual = true
			break
		}
	}
	if !textual && !(len(content) > 0 && (content[0] == '{' || content[0] == '[')) {
		return content
	}
	text := strings.ToValidUTF8(string(content), "�")
	if strings.Contains(contentType, "application/json") || (text != "" && (text[0] == '{' || text[0] == '[')) {
		var v any
		if err := json.Unmarshal(content, &v); err == nil {
			return v
		}
		return text
	}
	return text
}

var (
	durationRe    = regexp.MustCompile(`(\d+(?:\.\d+)?)(ms|h|m|s)`)
	durationUnits = map[string]float64{"ms": 0.001, "s": 1, "m": 60, "h": 3600}
)

func parseGoDuration(v string) (float64, bool) {
	v = strings.TrimSpace(v)
	parts := durationRe.FindAllStringSubmatch(v, -1)
	if len(parts) == 0 {
		return 0, false
	}
	var joined strings.Builder
	total := 0.0
	for _, p := range parts {
		joined.WriteString(p[1] + p[2])
		n, _ := strconv.ParseFloat(p[1], 64)
		total += n * durationUnits[p[2]]
	}
	if joined.String() != v {
		return 0, false
	}
	return total, true
}

// RetryAfter is how long the server asked us to wait (retry-after-ms,
// retry-after seconds or HTTP date, else OpenAI's x-ratelimit-reset-* for an
// exhausted window). ok is false when there is no usable hint.
func RetryAfter(h http.Header) (time.Duration, bool) {
	secs := func(f float64) time.Duration { return time.Duration(math.Max(0, f) * float64(time.Second)) }
	if ms := h.Get("retry-after-ms"); ms != "" {
		if f, err := strconv.ParseFloat(ms, 64); err == nil {
			return secs(f / 1000), true
		}
	}
	if v := h.Get("retry-after"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return secs(f), true
		}
		if t, err := http.ParseTime(v); err == nil {
			return secs(time.Until(t).Seconds()), true
		}
	}
	var exhausted, resets []float64
	sawRemaining := false
	for _, kind := range []string{"requests", "tokens"} {
		if h.Get("x-ratelimit-remaining-"+kind) != "" || len(h.Values("x-ratelimit-remaining-"+kind)) > 0 {
			sawRemaining = true
		}
		raw := h.Get("x-ratelimit-reset-" + kind)
		if raw == "" {
			continue
		}
		s, ok := parseGoDuration(raw)
		if !ok {
			continue
		}
		resets = append(resets, s)
		if rem := h.Values("x-ratelimit-remaining-" + kind); len(rem) > 0 && strings.TrimSpace(rem[0]) == "0" {
			exhausted = append(exhausted, s)
		}
	}
	if len(exhausted) > 0 {
		m := exhausted[0]
		for _, e := range exhausted[1:] {
			m = math.Max(m, e)
		}
		return secs(m), true
	}
	if len(resets) > 0 && !sawRemaining {
		m := resets[0]
		for _, r := range resets[1:] {
			m = math.Min(m, r)
		}
		return secs(m), true
	}
	return 0, false
}

// MergeURL joins a provider api_base with a relative endpoint.
func MergeURL(base, u string) string {
	if strings.Contains(u, "://") {
		return u
	}
	return strings.TrimRight(base, "/") + "/" + strings.TrimLeft(u, "/")
}

// ParseErrorBody extracts a provider error message (pyllym parse_error_body).
func ParseErrorBody(body any) string {
	if s, ok := body.(string); ok {
		var v any
		if err := json.Unmarshal([]byte(s), &v); err != nil {
			return s
		}
		body = v
	}
	if b, ok := body.([]byte); ok {
		var v any
		if err := json.Unmarshal(b, &v); err != nil {
			return string(b)
		}
		body = v
	}
	switch b := body.(type) {
	case map[string]any:
		if e, ok := b["error"].(string); ok {
			return e
		}
		if em, ok := b["error"].(map[string]any); ok {
			if s, ok := em["message"].(string); ok {
				return s
			}
		}
		if s, ok := b["message"].(string); ok {
			return s
		}
		if s, ok := b["detail"].(string); ok {
			return s
		}
		return ""
	case []any:
		var parts []string
		for _, p := range b {
			pm, ok := p.(map[string]any)
			if !ok {
				continue
			}
			if e, ok := pm["error"].(string); ok && e != "" {
				parts = append(parts, e)
			} else if em, ok := pm["error"].(map[string]any); ok {
				if s, ok := em["message"].(string); ok && s != "" {
					parts = append(parts, s)
				}
			}
		}
		return strings.Join(parts, ". ")
	case nil:
		return ""
	}
	return fmt.Sprint(body)
}

// Endpoint is what a Connection needs from its provider.
type Endpoint interface {
	APIBase() (string, error)
	Headers() map[string]string
	ParseError(*Response) string
}

// Multipart is a multipart/form-data payload.
type Multipart struct {
	Data  map[string]string
	Files map[string]MultipartFile
}

// MultipartFile is one uploaded file part.
type MultipartFile struct {
	Filename string
	Content  []byte
}

// Connection sends one provider's requests with retries.
type Connection struct {
	endpoint Endpoint
	config   *Config
	// Sleep is replaceable in tests.
	Sleep func(context.Context, time.Duration) error
	// Client overrides the shared HTTP client (tests, custom transports).
	Client *http.Client
}

// NewConnection binds a provider to a config.
func NewConnection(e Endpoint, cfg *Config) *Connection {
	return &Connection{endpoint: e, config: cfg, Sleep: sleepCtx}
}

// Config returns the connection's configuration.
func (c *Connection) Config() *Config { return c.config }

func sleepCtx(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

var (
	clientMu    sync.Mutex
	clientCache = map[string]*http.Client{}
)

// sharedClient pools connections per proxy setting.
func sharedClient(proxy string) *http.Client {
	clientMu.Lock()
	defer clientMu.Unlock()
	if c, ok := clientCache[proxy]; ok {
		return c
	}
	tr := http.DefaultTransport.(*http.Transport).Clone()
	if proxy != "" {
		if u, err := url.Parse(proxy); err == nil {
			tr.Proxy = http.ProxyURL(u)
		}
	}
	c := &http.Client{Transport: tr}
	clientCache[proxy] = c
	return c
}

func (c *Connection) client() *http.Client {
	if c.Client != nil {
		return c.Client
	}
	return sharedClient(c.config.HTTPProxy)
}

// Post sends a JSON (or Multipart) payload.
func (c *Connection) Post(ctx context.Context, u string, payload any, headers map[string]string) (*Response, error) {
	return c.request(ctx, http.MethodPost, u, payload, headers)
}

// Get sends a GET.
func (c *Connection) Get(ctx context.Context, u string, headers map[string]string) (*Response, error) {
	return c.request(ctx, http.MethodGet, u, nil, headers)
}

// Delete sends a DELETE.
func (c *Connection) Delete(ctx context.Context, u string, headers map[string]string) (*Response, error) {
	return c.request(ctx, http.MethodDelete, u, nil, headers)
}

func (c *Connection) build(ctx context.Context, method, u string, payload any, headers map[string]string) (*http.Request, error) {
	base, err := c.endpoint.APIBase()
	if err != nil {
		return nil, err
	}
	var raw []byte
	contentType := ""
	switch p := payload.(type) {
	case nil:
	case *Multipart:
		var buf bytes.Buffer
		w := multipart.NewWriter(&buf)
		for _, k := range sortedStringKeys(p.Data) {
			_ = w.WriteField(k, p.Data[k])
		}
		for _, name := range sortedFileKeys(p.Files) {
			f := p.Files[name]
			fw, err := w.CreateFormFile(name, f.Filename)
			if err != nil {
				return nil, err
			}
			_, _ = fw.Write(f.Content)
		}
		_ = w.Close()
		raw, contentType = buf.Bytes(), w.FormDataContentType()
	case []byte:
		raw, contentType = p, "application/octet-stream"
	default:
		b, err := MarshalPayload(p)
		if err != nil {
			return nil, err
		}
		raw, contentType = b, "application/json"
	}
	var body io.Reader
	if raw != nil {
		body = bytes.NewReader(raw)
	}
	req, err := http.NewRequestWithContext(ctx, method, MergeURL(base, u), body)
	if err != nil {
		return nil, err
	}
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	for k, v := range c.endpoint.Headers() {
		if v != "" {
			req.Header.Set(k, v)
		}
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	if signer, ok := c.endpoint.(RequestSigner); ok {
		if err := signer.Sign(req, raw); err != nil {
			return nil, err
		}
	}
	return req, nil
}

// MarshalPayload encodes a JSON payload without HTML escaping (Python's
// json.dumps never escapes <, > or &).
func MarshalPayload(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimSuffix(buf.Bytes(), []byte("\n")), nil
}

func sortedFileKeys(m map[string]MultipartFile) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sortStrings(out)
	return out
}

func (c *Connection) request(ctx context.Context, method, u string, payload any, headers map[string]string) (*Response, error) {
	var last error
	for attempt := 0; attempt <= c.config.MaxRetries; attempt++ {
		resp, err := c.once(ctx, method, u, payload, headers)
		if err == nil {
			return resp, nil
		}
		if !retryable(err) {
			return nil, err
		}
		last = err
		if ok, werr := c.waitBeforeRetry(ctx, attempt, err, method, u); werr != nil {
			return nil, werr
		} else if !ok {
			break
		}
	}
	return nil, finalError(last)
}

func (c *Connection) once(ctx context.Context, method, u string, payload any, headers map[string]string) (*Response, error) {
	actx, cancel := ctx, context.CancelFunc(func() {})
	if c.config.RequestTimeout > 0 {
		actx, cancel = context.WithTimeout(ctx, c.config.RequestTimeout)
	}
	defer cancel()
	req, err := c.build(actx, method, u, payload, headers)
	if err != nil {
		return nil, err
	}
	httpResp, err := c.client().Do(req)
	if err != nil {
		return nil, transportError(ctx, err)
	}
	defer httpResp.Body.Close()
	content, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, transportError(ctx, err)
	}
	resp := &Response{
		Status:  httpResp.StatusCode,
		Body:    ParseBody(httpResp.Header.Get("Content-Type"), content),
		Headers: httpResp.Header,
		Content: content,
	}
	if resp.Status >= 400 {
		return nil, c.errorFor(resp)
	}
	return resp, nil
}

func (c *Connection) errorFor(resp *Response) *Error {
	message := c.endpoint.ParseError(resp)
	e := NewError(ErrorKindForStatus(resp.Status, message), resp, message)
	if d, ok := RetryAfter(resp.Headers); ok {
		e.RetryAfter = &d
	}
	return e
}

// retryableTransport marks transport failures pyllym retries (timeouts,
// connection and payload errors), as opposed to bad URLs or TLS failures.
type retryableTransport struct{ err *Error }

func (r retryableTransport) Error() string { return r.err.Error() }
func (r retryableTransport) Unwrap() error { return r.err }

func transportError(ctx context.Context, err error) error {
	if ctx.Err() != nil {
		// The caller's own deadline/cancel: never retried.
		return WrapConnectionError(err)
	}
	var ne net.Error
	var oe *net.OpError
	if errors.As(err, &ne) && ne.Timeout() || errors.As(err, &oe) || errors.Is(err, io.ErrUnexpectedEOF) ||
		errors.Is(err, io.EOF) || errors.Is(err, context.DeadlineExceeded) || strings.Contains(err.Error(), "connection reset") ||
		strings.Contains(err.Error(), "connection refused") {
		return retryableTransport{WrapConnectionError(err)}
	}
	return WrapConnectionError(err)
}

func retryable(err error) bool {
	if _, ok := err.(retryableTransport); ok {
		return true
	}
	var e *Error
	return errors.As(err, &e) && e.Kind.Retryable()
}

func finalError(err error) error {
	if rt, ok := err.(retryableTransport); ok {
		return rt.err
	}
	return err
}

// waitBeforeRetry sleeps out the backoff; false means give up.
func (c *Connection) waitBeforeRetry(ctx context.Context, attempt int, err error, method, u string) (bool, error) {
	if attempt >= c.config.MaxRetries {
		return false, nil
	}
	delay, ok := c.RetryDelay(attempt, err)
	if !ok {
		slog.Warn("Not retrying: server asked to wait above RetryMaxInterval", "method", method, "url", u)
		return false, nil
	}
	slog.Info("Retrying", "method", method, "url", u, "delay", delay, "attempt", attempt+1, "max", c.config.MaxRetries, "err", err)
	if err := c.Sleep(ctx, delay); err != nil {
		return false, WrapConnectionError(err)
	}
	return true, nil
}

// RetryDelay is the wait before retry attempt+1 (pyllym _retry_delay). A
// Retry-After hint wins; one above RetryMaxInterval means don't retry.
func (c *Connection) RetryDelay(attempt int, err error) (time.Duration, bool) {
	cfg := c.config
	jitter := cfg.RetryIntervalRandomness * rand.Float64()
	var e *Error
	if errors.As(err, &e) && e.RetryAfter != nil {
		hint := *e.RetryAfter
		if cfg.RetryMaxInterval > 0 && hint > cfg.RetryMaxInterval {
			return 0, false
		}
		return hint + time.Duration(float64(cfg.RetryInterval)*jitter), true
	}
	base := cfg.RetryInterval
	if e != nil && e.Kind == KindRateLimit {
		base = cfg.RateLimitRetryInterval
	}
	interval := float64(base) * math.Pow(cfg.RetryBackoffFactor, float64(attempt))
	delay := time.Duration(interval + interval*jitter)
	if cfg.RetryMaxInterval > 0 && delay > cfg.RetryMaxInterval {
		delay = cfg.RetryMaxInterval
	}
	return delay, true
}

// Stream POSTs payload and calls onData for each body chunk. Retryable
// failures are retried only before the first byte arrives. RequestTimeout is
// an idle timeout between reads, so long streams are not cut off.
func (c *Connection) Stream(ctx context.Context, u string, payload any, headers map[string]string, onData func([]byte) error) error {
	var last error
	for attempt := 0; attempt <= c.config.MaxRetries; attempt++ {
		started, err := c.streamOnce(ctx, u, payload, headers, onData)
		if err == nil {
			return nil
		}
		if started || !retryable(err) {
			return finalError(err)
		}
		last = err
		if ok, werr := c.waitBeforeRetry(ctx, attempt, err, http.MethodPost, u); werr != nil {
			return werr
		} else if !ok {
			break
		}
	}
	return finalError(last)
}

func (c *Connection) streamOnce(ctx context.Context, u string, payload any, headers map[string]string, onData func([]byte) error) (bool, error) {
	sctx, cancel := context.WithCancel(ctx)
	defer cancel()
	req, err := c.build(sctx, http.MethodPost, u, payload, headers)
	if err != nil {
		return false, err
	}
	var idle *time.Timer
	if c.config.RequestTimeout > 0 {
		idle = time.AfterFunc(c.config.RequestTimeout, cancel)
		defer idle.Stop()
	}
	httpResp, err := c.client().Do(req)
	if err != nil {
		return false, transportError(ctx, err)
	}
	defer httpResp.Body.Close()
	if httpResp.StatusCode >= 400 {
		content, _ := io.ReadAll(httpResp.Body)
		resp := &Response{Status: httpResp.StatusCode, Body: ParseBody(httpResp.Header.Get("Content-Type"), content), Headers: httpResp.Header, Content: content}
		return false, c.errorFor(resp)
	}
	buf := make([]byte, 32*1024)
	started := false
	for {
		if idle != nil {
			idle.Reset(c.config.RequestTimeout)
		}
		n, rerr := httpResp.Body.Read(buf)
		if n > 0 {
			started = true
			if err := onData(append([]byte(nil), buf[:n]...)); err != nil {
				return true, err
			}
		}
		if rerr == io.EOF {
			return started, nil
		}
		if rerr != nil {
			return started, transportError(ctx, rerr)
		}
	}
}

func sortedStringKeys(m map[string]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sortStrings(out)
	return out
}
