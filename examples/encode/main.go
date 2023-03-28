package main

import (
	"dagobert"
	"encoding/binary"
	"fmt"
	"log"

	"gonum.org/v1/gonum/mat"
)

func fn1() {
	client, err := dagobert.NewClient("grpc://localhost:51000")
	if err != nil {
		log.Fatal(err)
	}

	docs, err := client.Encode(
		[]*dagobert.Document{
			dagobert.NewTextDocument("我 看到 水果"),
			dagobert.NewTextDocument("我 看到 水车"),
			dagobert.NewTextDocument("我 看到 苹果"),
			dagobert.NewTextDocument("我 看到 的 就是 苹果"),
			dagobert.NewTextDocument("我 看不到 水果"),
			dagobert.NewTextDocument("不过 考虑 到 我们 的 帮助 手册 及 FAQ 总 字数 已经 超过 30 万字   而   gpt - 3.5 - turbo   每次 对话 最 多 只能 支持   4096   个   Token   ( 经过 我们 的 测试   这 大概 是 三千多 个 简中字 )   还 需要 考虑 客户 的 问题   回答 和 系统配置 占用 的 字符 . 所以 我们 需要 一个 办法 来 让   gpt - 3.5 - turbo   可以 在   2000   字 左右 时 了解 我们 的  帮助 手册"),
		},
	)
	if err != nil {
		log.Fatal(err)
	}

	var vets []mat.Vector

	var fltValues [][]float64

	for _, d := range docs {
		num := int(d.GetEmbedding().GetDense().GetShape()[0])
		bytes := d.GetEmbedding().GetDense().GetBuffer()
		step := len(bytes) / num
		// step = 8
		var farr []float64
		for i := step; i <= len(bytes); i += step {
			btarr := bytes[i-step : i]
			// bits := binary.BigEndian.Uint32(btarr)
			i64val := binary.LittleEndian.Uint32(btarr)
			// i64val := binary.LittleEndian.Uint64(btarr)
			farr = append(farr, float64(i64val))
			// farr = append(farr, float64(math.Float32frombits(bits)))
			// bits := binary.BigEndian.Uint64(btarr)
			// farr = append(farr, math.Float64frombits(bits))
		}

		// fmt.Println(len(bytes))
		// for i := 0; i < len(bytes); i++ {
		// 	farr = append(farr, float64(bytes[i]))
		// }

		vec := mat.NewVecDense(len(farr), farr)
		vets = append(vets, vec)

		fltValues = append(fltValues, farr)
	}

	// for _, doc := range docs {
	// 	fmt.Println(doc.GetEmbedding().GetDense().GetBuffer())
	// }

	// for _, vet := range vets {
	// 	fmt.Println(vet)
	// }

	for i, _ := range vets {
		// fmt.Printf("vets sim 0 -> %d : %f\n", i, dagobert.CosSim(vets[0], vets[i]))
		v, _ := dagobert.Cosine(fltValues[0], fltValues[i])
		fmt.Printf("float sim 0 -> %d : %f\n", i, v)
	}

	fmt.Println(docs[0].GetEmbedding().GetDense().GetShape())
}

func fn2() {
	client, err := dagobert.NewClient("grpc://localhost:51000")
	if err != nil {
		log.Fatal(err)
	}

	docs, err := client.Encode(
		[]*dagobert.Document{
			dagobert.NewTextDocument("我 看到 水果"),
			dagobert.NewTextDocument("我 看到 水车"),
			dagobert.NewTextDocument("我 看到 苹果"),
			dagobert.NewTextDocument("我 看到 的 就是 苹果"),
			dagobert.NewTextDocument("我 看不到 水果"),
			dagobert.NewTextDocument("今天 下午 发生 了 一件 非常 奇怪 的 事情"),
		},
	)
	if err != nil {
		log.Fatal(err)
	}

	var vets []mat.Vector

	for _, d := range docs {
		byteCount := int(d.GetEmbedding().GetDense().GetShape()[0])
		bytes := d.GetEmbedding().GetDense().GetBuffer()
		step := len(bytes) / byteCount
		// step = 8
		var row = make([][]float32, byteCount)
		for i, g := step, 0; i <= len(bytes); i, g = i+step, g+1 {
			btarr := bytes[i-step : i]
			var col = make([]float32, byteCount)
			for ii, b := range btarr {
				col[ii] = float32(b)
			}
			row[g] = col
		}

		vec := dagobert.PoolVector(row)
		vets = append(vets, vec)
	}

	// for _, doc := range docs {
	// 	fmt.Println(doc.GetEmbedding().GetDense().GetBuffer())
	// }

	// for _, vet := range vets {
	// 	fmt.Println(vet)
	// }

	for i, _ := range vets {
		fmt.Printf("vets sim 0 -> %d : %f\n", i, dagobert.CosSim(vets[0], vets[i]))
	}

	fmt.Println(docs[0].GetEmbedding().GetDense().GetShape())
}

func main() {
	fn1()
}
