// Función para obtener los datos de los coches
async function getData() {
  const response = await fetch(
    "https://storage.googleapis.com/tfjs-tutorials/carsData.json"
  );
  const data = await response.json();
  return data
    .filter((car) => car.Miles_per_Gallon !== null && car.Horsepower !== null)
    .map((car) => ({ mpg: car.Miles_per_Gallon, horsepower: car.Horsepower }));
}

// Función para crear el modelo
function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));
  model.add(tf.layers.dense({ units: 1, useBias: true }));
  return model;
}

// Función para convertir datos a tensores
function convertToTensor(data) {
  const inputs = data.map((d) => d.horsepower);
  const labels = data.map((d) => d.mpg);
  const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
  const labelTensor = tf.tensor2d(labels, [labels.length, 1]);
  const inputMax = inputTensor.max();
  const inputMin = inputTensor.min();
  const labelMax = labelTensor.max();
  const labelMin = labelTensor.min();
  const normalizedInputs = inputTensor
    .sub(inputMin)
    .div(inputMax.sub(inputMin));
  const normalizedLabels = labelTensor
    .sub(labelMin)
    .div(labelMax.sub(labelMin));
  return {
    inputs: normalizedInputs,
    labels: normalizedLabels,
    inputMax,
    inputMin,
    labelMax,
    labelMin,
  };
}

// Función para entrenar el modelo
async function trainModel(model, inputs, labels) {
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ["mse"],
  });
  const batchSize = 32;
  const epochs = 50;
  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: "Training Performance" },
      ["loss", "mse"],
      { height: 200, callbacks: ["onEpochEnd"] }
    ),
  });
}

// Función para probar el modelo
function testModel(model, inputData, normalizationData) {
  const { inputMax, inputMin, labelMin, labelMax } = normalizationData;
  const xs = tf.linspace(0, 1, 100);
  const preds = model.predict(xs.reshape([100, 1]));
  const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);
  const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);
  const predictedPoints = Array.from(unNormXs.dataSync()).map((val, i) => ({
    x: val,
    y: unNormPreds.dataSync()[i],
  }));
  const originalPoints = inputData.map((d) => ({ x: d.horsepower, y: d.mpg }));
  tfvis.render.scatterplot(
    { name: "Model Predictions vs Original Data" },
    {
      values: [originalPoints, predictedPoints],
      series: ["original", "predicted"],
    },
    { xLabel: "Horsepower", yLabel: "MPG", height: 300 }
  );
}

// Función principal para ejecutar el código
async function run() {
  const data = await getData();
  const tensorData = convertToTensor(data);
  const model = createModel();
  await trainModel(model, tensorData.inputs, tensorData.labels);
  testModel(model, data, tensorData);
}

// Ejecutar la función principal cuando se carga el documento
document.addEventListener("DOMContentLoaded", run);
