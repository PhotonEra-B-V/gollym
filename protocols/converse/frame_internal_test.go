package converse

import (
	"bytes"
	"errors"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"

	"github.com/PhotonEra-B-V/gollym/core"
)

func encodeFrames(t *testing.T, payloads ...string) []byte {
	t.Helper()
	var buf bytes.Buffer
	for _, p := range payloads {
		msg := eventstream.Message{Payload: []byte(p)}
		msg.Headers.Set(":message-type", eventstream.StringValue("event"))
		msg.Headers.Set(":event-type", eventstream.StringValue("contentBlockDelta"))
		if err := eventstream.NewEncoder().Encode(&buf, msg); err != nil {
			t.Fatal(err)
		}
	}
	return buf.Bytes()
}

// Frames split across arbitrary read boundaries are reassembled.
func TestFrameReaderReassemblesSplitFrames(t *testing.T) {
	data := encodeFrames(t, `{"delta":{"text":"a"}}`, `{"delta":{"text":"bc"}}`, `{"delta":{"text":"d"}}`)
	for _, step := range []int{1, 3, 7, 50, len(data)} {
		fr := &frameReader{dec: eventstream.NewDecoder()}
		var got []string
		for i := 0; i < len(data); i += step {
			end := min(i+step, len(data))
			err := fr.write(data[i:end], func(m eventstream.Message) error {
				ev, err := eventFor(m)
				if err != nil {
					return err
				}
				got = append(got, core.DigString(ev, "contentBlockDelta", "delta", "text"))
				return nil
			})
			if err != nil {
				t.Fatalf("step %d: %v", step, err)
			}
		}
		if len(fr.buf) != 0 || len(got) != 3 || got[0] != "a" || got[1] != "bc" || got[2] != "d" {
			t.Fatalf("step %d: got %v, %d bytes left", step, got, len(fr.buf))
		}
	}
}

func TestFrameReaderRejectsCorruptFrames(t *testing.T) {
	data := encodeFrames(t, `{"delta":{"text":"a"}}`)
	data[len(data)-6] ^= 0xff // payload byte: message CRC mismatch
	fr := &frameReader{dec: eventstream.NewDecoder()}
	err := fr.write(data, func(eventstream.Message) error { return nil })
	var e *core.Error
	if !errors.As(err, &e) {
		t.Fatalf("err %v", err)
	}
	fr = &frameReader{dec: eventstream.NewDecoder()}
	if err := fr.write([]byte{0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0}, func(eventstream.Message) error { return nil }); err == nil {
		t.Fatal("oversized frame accepted")
	}
}
