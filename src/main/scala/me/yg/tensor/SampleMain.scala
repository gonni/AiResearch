package me.yg.tensor

import org.platanios.tensorflow.api.Shape
import org.platanios.tensorflow.api.tensors.Tensor

object SampleMain {
  def main(args: Array[String]): Unit = {
    println("Active AI on Scala")
    val tensor = Tensor.zeros[Int](Shape(2, 5))
    tensor.summarize()
  }
}
