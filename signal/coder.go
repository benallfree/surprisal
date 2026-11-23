package signal

import "sort"

// Witten-Neal-Cleary arithmetic coder, 32-bit (SPEC.md §5).
const (
	top  = 1<<32 - 1
	half = 1 << 31
	qtr  = 1 << 30
)

type encoder struct {
	low, high, pending uint64
	out                []byte
	acc, nb            uint8
}

func newEncoder() *encoder { return &encoder{high: top} }

func (e *encoder) bit(b uint8) {
	e.acc = e.acc<<1 | b
	e.nb++
	if e.nb == 8 {
		e.out = append(e.out, e.acc)
		e.acc, e.nb = 0, 0
	}
}

func (e *encoder) emit(b uint8) {
	e.bit(b)
	for ; e.pending > 0; e.pending-- {
		e.bit(1 - b)
	}
}

// code narrows the interval to [lo, hi) out of total.
func (e *encoder) code(total, hi, lo uint64) {
	rng := e.high - e.low + 1
	e.high = e.low + rng*hi/total - 1
	e.low = e.low + rng*lo/total
	for {
		switch {
		case e.high < half:
			e.emit(0)
		case e.low >= half:
			e.emit(1)
			e.low -= half
			e.high -= half
		case e.low >= qtr && e.high < half+qtr:
			e.pending++
			e.low -= qtr
			e.high -= qtr
		default:
			return
		}
		e.low, e.high = 2*e.low, 2*e.high+1
	}
}

// codeToken codes token t from a cumulative count table.
func (e *encoder) codeToken(cum []int64, t int) {
	lo := int64(0)
	if t > 0 {
		lo = cum[t-1]
	}
	e.code(uint64(cum[len(cum)-1]), uint64(cum[t]), uint64(lo))
}

func (e *encoder) finish() []byte {
	e.pending++
	if e.low < qtr {
		e.emit(0)
	} else {
		e.emit(1)
	}
	for e.nb != 0 {
		e.bit(0)
	}
	return e.out
}

type decoder struct {
	data             []byte
	pos              int
	low, high, value uint64
}

func newDecoder(data []byte) *decoder {
	d := &decoder{data: data, high: top}
	for i := 0; i < 32; i++ {
		d.value = 2*d.value + d.bit()
	}
	return d
}

func (d *decoder) bit() uint64 {
	i := d.pos
	d.pos++
	if i>>3 >= len(d.data) {
		return 0
	}
	return uint64(d.data[i>>3]>>(7-uint(i&7))) & 1
}

// next decodes one token from a cumulative count table.
func (d *decoder) next(cum []int64) int {
	total := uint64(cum[len(cum)-1])
	rng := d.high - d.low + 1
	target := int64(((d.value-d.low+1)*total - 1) / rng)
	t := sort.Search(len(cum), func(i int) bool { return cum[i] > target })
	hi, lo := uint64(cum[t]), uint64(0)
	if t > 0 {
		lo = uint64(cum[t-1])
	}
	d.high = d.low + rng*hi/total - 1
	d.low = d.low + rng*lo/total
	for {
		switch {
		case d.high < half:
		case d.low >= half:
			d.value -= half
			d.low -= half
			d.high -= half
		case d.low >= qtr && d.high < half+qtr:
			d.value -= qtr
			d.low -= qtr
			d.high -= qtr
		default:
			return t
		}
		d.low, d.high, d.value = 2*d.low, 2*d.high+1, 2*d.value+d.bit()
	}
}
