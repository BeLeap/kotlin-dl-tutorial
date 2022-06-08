package codes.beleap.kotlin_dl_tutorial

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.layer.core.*
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.*
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.summary.printSummary
import org.jetbrains.kotlinx.dl.dataset.fashionMnist
import org.jetbrains.kotlinx.dl.api.core.WritingMode
import org.jetbrains.kotlinx.dl.api.inference.TensorFlowInferenceModel
import java.io.File

fun main() {
  val stringLabels =
      mapOf(
          0 to "T-shirt/top",
          1 to "Trouser",
          2 to "Pullover",
          3 to "Dress",
          4 to "Coat",
          5 to "Sandal",
          6 to "Shirt",
          7 to "Sneaker",
          8 to "Bag",
          9 to "Ankle boot",
      )

  val model =
      Sequential.of(
          Input(28, 28, 1),
          Flatten(),
          Dense(300),
          Dense(100),
          Dense(10),
      )

  val (train, test) = fashionMnist()

  model.use {
    it.compile(
        optimizer = Adam(),
        loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
        metric = Metrics.ACCURACY,
    )

    it.printSummary()

    it.fit(
        dataset = train,
        epochs = 10,
        batchSize = 100,
    )

    val accuracy = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

    println("Accuracy: $accuracy")
    it.save(File("model/my_model"), writingMode = WritingMode.OVERRIDE)
  }

  TensorFlowInferenceModel.load(File("model/my_model")).use {
    it.reshape(28, 28, 1)
    val prediction = it.predict(test.getX(0))
    val actualLabel = test.getY(0)

    println("Predicted label is: $prediction. This corresponds to class ${stringLabels[prediction]}")
    println("Actual label is: $actualLabel")
  }
}

