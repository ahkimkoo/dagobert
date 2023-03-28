package dagobert

import (
	"errors"
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

func PoolVector(toks [][]float32) mat.Vector {
	c := len(toks[0])
	vec := mat.NewVecDense(c, nil)
	x := make([]float64, c)
	for i := range x {
		for j, tok := range toks {
			x[j] = float64(tok[i])
		}
		vec.SetVec(i, stat.Mean(x, nil))
	}
	return vec
}

func CosSim(x, y mat.Vector) float64 {
	// return mat.Dot(x, y)
	return (mat.Dot(x, y)) / (mat.Norm(x, 2) * mat.Norm(y, 2))
}

func Cosine(a []float64, b []float64) (cosine float64, err error) {
	count := 0
	length_a := len(a)
	length_b := len(b)
	if length_a > length_b {
		count = length_a
	} else {
		count = length_b
	}
	sumA := 0.0
	s1 := 0.0
	s2 := 0.0
	for k := 0; k < count; k++ {
		if k >= length_a {
			s2 += math.Pow(b[k], 2)
			continue
		}
		if k >= length_b {
			s1 += math.Pow(a[k], 2)
			continue
		}
		sumA += a[k] * b[k]
		s1 += math.Pow(a[k], 2)
		s2 += math.Pow(b[k], 2)
	}
	if s1 == 0 || s2 == 0 {
		return 0.0, errors.New("Vectors should not be null (all zeros)")
	}
	return sumA / (math.Sqrt(s1) * math.Sqrt(s2)), nil
}
