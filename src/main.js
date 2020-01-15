const { tf } = window;
import { IMAGENET_CLASSES, IMAGE_SIZE, MODEL_NAME } from "./constants.js";

(async () => {
  const $img = document.getElementById("img");
  const $res = document.getElementById("result");

  const urlParams = new URLSearchParams(location.search);
  if (!urlParams.has("nowasm")) {
    await tf.setBackend("wasm");
    console.log("Use wasm backend");
  } else {
    console.log("Use normal backend");
  }

  console.time("Load mobilenet");
  const mobilenet = await tf.loadLayersModel(
    `https://storage.googleapis.com/tfjs-models/tfjs/${MODEL_NAME}/model.json`
  );
  // warmup
  mobilenet.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();
  console.log("model: %s ready", MODEL_NAME);
  console.timeEnd("Load mobilenet");

  console.time("Convert image");
  const imgFloat = imgToFloat($img);
  console.timeEnd("Convert image");

  console.time("Perform prediction");
  const logits = predict(mobilenet, imgFloat);
  console.timeEnd("Perform prediction");

  console.time("Logits to values");
  const values = await logits.data();
  console.timeEnd("Logits to values");

  console.time("Display results");
  const classes = getTopKClassesByLogitsData(values, 10);
  $res.textContent = JSON.stringify(classes, null, 2);
  console.timeEnd("Display results");
})();

function imgToFloat($img) {
  return tf.tidy(() => {
    return tf.browser
      .fromPixels($img)
      .resizeBilinear([IMAGE_SIZE, IMAGE_SIZE])
      .expandDims(0)
      .toFloat();
  });
}

function predict(mobilenet, imgFloat) {
  return tf.tidy(() => {
    const offset = tf.scalar(127.5);
    // Normalize the image from [0, 255] to [-1, 1].
    const normalized = imgFloat.sub(offset).div(offset);
    // Reshape to a single-element batch so we can pass it to predict.
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
    return mobilenet.predict(batched);
  });
}

function getTopKClassesByLogitsData(values, topK) {
  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({ value: values[i], index: i });
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });

  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  const topClassesAndProbs = [];
  for (let i = 0; i < topkIndices.length; i++) {
    topClassesAndProbs.push({
      className: IMAGENET_CLASSES[topkIndices[i]],
      probability: topkValues[i]
    });
  }
  return topClassesAndProbs;
}
