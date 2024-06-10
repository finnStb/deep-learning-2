// script.js
let lossChart;

let trainClean, testClean, trainNoisy, testNoisy; // Globale Variablen für die Daten

let loaderCount = 0;

const ctxUnrauschedData = document.getElementById('chartUnrauschedData').getContext('2d');
const ctxRauschedData = document.getElementById('chartRauschedData').getContext('2d');
const ctxUnrauschedPredictionTrain = document.getElementById('chartUnrauschedPredictionTrain').getContext('2d');
const ctxUnrauschedPredictionTest = document.getElementById('chartUnrauschedPredictionTest').getContext('2d');
const ctxBestFitPredictionTrain = document.getElementById('chartBestFitPredictionTrain').getContext('2d');
const ctxBestFitPredictionTest = document.getElementById('chartBestFitPredictionTest').getContext('2d');
const ctxOverfitPredictionTrain = document.getElementById('chartOverfitPredictionTrain').getContext('2d');
const ctxOverfitPredictionTest = document.getElementById('chartOverfitPredictionTest').getContext('2d');

// Farben für die Datensätze
const trainColor = 'rgba(75, 192, 192, 0.8)';
const testColor = 'rgba(255, 99, 132, 0.8)';

// data.js

function generateGaussianNoise(mean = 0, variance = 1) {
    let u1 = Math.random();
    let u2 = Math.random();
    let randStdNormal = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2); // Box-Muller transform
    return mean + Math.sqrt(variance) * randStdNormal;
}

// Daten ohne Rauschen generieren
let dataClean = generateData(100);

// Daten aufteilen
({ trainData: trainClean, testData: testClean } = splitData(dataClean));

// Daten mit Rauschen hinzufügen
trainNoisy = addNoise(trainClean);
testNoisy = addNoise(testClean);

// Berechnung der Varianz des hinzugefügten Rauschens
const noiseValues = [];
for (let i = 0; i < 10000; i++) {
    noiseValues.push(generateGaussianNoise(0, 0.05));
}

const mean = noiseValues.reduce((acc, val) => acc + val, 0) / noiseValues.length;
const variance = noiseValues.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / noiseValues.length;

console.log(`Mean: ${mean}, Variance: ${variance}`);

async function updateData() {
    const dataPointsInput = document.getElementById('dataPoints');
    const noiseVarianceInput = document.getElementById('noiseVariance');

    const N = parseInt(dataPointsInput.value);
    const variance = parseFloat(noiseVarianceInput.value);

    // Daten ohne Rauschen generieren
    dataClean = generateData(N);

    // Daten aufteilen
    ({ trainData: trainClean, testData: testClean } = splitData(dataClean));

    // Daten mit Rauschen hinzufügen
    trainNoisy = addNoise(trainClean, variance);
    testNoisy = addNoise(testClean, variance);

    // Diagramme aktualisieren
    plotData(ctxUnrauschedData, trainClean, testClean);
    plotData(ctxRauschedData, trainNoisy, testNoisy);

    // Trainings neu starten
    retrainModel('unrausched');
    retrainModel('bestFit');
    retrainModel('overfit');
}

function generateData(N) {
    const xValues = [];
    const yValues = [];
    for (let i = 0; i < N; i++) {
        const x = Math.random() * 4 - 2; // Zufälliger Wert zwischen -2 und 2
        let y = 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1;
        xValues.push(x);
        yValues.push(y);
    }
    return { xValues, yValues };
}

function addNoise(data, variance = 0.05) {
    const noisyYValues = data.yValues.map(y => y + generateGaussianNoise(0, variance));
    return { xValues: data.xValues, yValues: noisyYValues };
}

function splitData(data) {
    const N = data.xValues.length;
    const indices = Array.from(Array(N).keys());
    indices.sort(() => Math.random() - 0.5); // Zufälliges Mischen der Indizes

    const trainSize = Math.floor(N / 2);
    const trainIndices = indices.slice(0, trainSize);
    const testIndices = indices.slice(trainSize);

    const trainData = {
        xValues: trainIndices.map(i => data.xValues[i]),
        yValues: trainIndices.map(i => data.yValues[i]),
    };

    const testData = {
        xValues: testIndices.map(i => data.xValues[i]),
        yValues: testIndices.map(i => data.yValues[i]),
    };

    return { trainData, testData };
}

// Daten plotten
function plotData(ctx, trainData, testData) {
    if (ctx.chart) {
        ctx.chart.destroy();
    }

    ctx.chart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: `Training data`,
                    data: trainData.xValues.map((x, i) => ({ x: x, y: trainData.yValues[i] })),
                    backgroundColor: trainColor,
                },
                {
                    label: `Test data`,
                    data: testData.xValues.map((x, i) => ({ x: x, y: testData.yValues[i] })),
                    backgroundColor: testColor,
                },
            ],
        },
        options: {
            scales: {
                x: { type: 'linear', position: 'bottom' },
                y: { beginAtZero: true },
            },
        },
    });
}

// Daten plotten
plotData(ctxUnrauschedData, trainClean, testClean, 'Unverrauschte Daten');
plotData(ctxRauschedData, trainNoisy, testNoisy, 'Verrauschte Daten');

// Modell erstellen
function createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 100, activation: 'relu', inputShape: [1] }));
    model.add(tf.layers.dense({ units: 100, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1 }));
    model.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });
    return model;
}

function showLoader() {
    document.getElementById('loader').classList.remove('hidden');
    document.getElementById('navbar').classList.add('animated');
    loaderCount++;
}

function hideLoader() {
    loaderCount--;
    if (loaderCount === 0)
        document.getElementById('loader').classList.add('hidden');
    document.getElementById('navbar').classList.remove('animated');
}

async function retrainModel(modelType) {
    showLoader();

    const epochsInput = document.getElementById(`${modelType}Epochs`);
    const learningRateInput = document.getElementById(`${modelType}LearningRate`);
    const batchSizeInput = document.getElementById(`${modelType}BatchSize`);
    const hiddenLayersInput = document.getElementById(`${modelType}HiddenLayers`);
    const neuronsPerLayerInput = document.getElementById(`${modelType}NeuronsPerLayer`);
    const activationInput = document.getElementById(`${modelType}Activation`);

    const epochs = parseInt(epochsInput.value);
    const learningRate = parseFloat(learningRateInput.value);
    const batchSize = parseInt(batchSizeInput.value);
    const hiddenLayers = parseInt(hiddenLayersInput.value);
    const neuronsPerLayer = parseInt(neuronsPerLayerInput.value);
    const activationFunction = activationInput.value;

    let model;
    let history;
    let trainData;
    let testData;
    let trainCtx;
    let testCtx;
    let lossCtx;
    let trainLossElementId;
    let testLossElementId;
    let color;

    if (modelType === 'unrausched') {
        trainData = trainClean;
        testData = testClean;
        trainCtx = ctxUnrauschedPredictionTrain;
        testCtx = ctxUnrauschedPredictionTest;
        lossCtx = document.getElementById('chartUnrauschedLossHistory').getContext('2d');
        trainLossElementId = 'lossUnrauschedTrain';
        testLossElementId = 'lossUnrauschedTest';
        color = trainColor;
    } else if (modelType === 'bestFit') {
        trainData = trainNoisy;
        testData = testNoisy;
        trainCtx = ctxBestFitPredictionTrain;
        testCtx = ctxBestFitPredictionTest;
        lossCtx = document.getElementById('chartBestFitLossHistory').getContext('2d');
        trainLossElementId = 'lossBestFitTrain';
        testLossElementId = 'lossBestFitTest';
        color = trainColor;
    } else if (modelType === 'overfit') {
        trainData = trainNoisy;
        testData = testNoisy;
        trainCtx = ctxOverfitPredictionTrain;
        testCtx = ctxOverfitPredictionTest;
        lossCtx = document.getElementById('chartOverfitLossHistory').getContext('2d');
        trainLossElementId = 'lossOverfitTrain';
        testLossElementId = 'lossOverfitTest';
        color = trainColor;
    }

    model = createModelWithCustomParameters(learningRate, hiddenLayers, neuronsPerLayer, activationFunction);

    history = await trainModel(model, trainData, testData, epochs, batchSize);
    plotPrediction(trainCtx, model, trainData, color, `Vorhersage mit ${modelType} Modell (Train)`, trainLossElementId);
    plotPrediction(testCtx, model, testData, testColor, `Vorhersage mit ${modelType} Modell (Test)`, testLossElementId);
    plotFinalLossHistory(lossCtx, history);

    hideLoader();
}

function createModelWithCustomParameters(learningRate, hiddenLayers, neuronsPerLayer, activationFunction) {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: neuronsPerLayer, activation: activationFunction, inputShape: [1] }));
    for (let i = 1; i < hiddenLayers; i++) {
        model.add(tf.layers.dense({ units: neuronsPerLayer, activation: activationFunction }));
    }
    model.add(tf.layers.dense({ units: 1 }));
    model.compile({ optimizer: tf.train.adam(learningRate), loss: 'meanSquaredError' });
    return model;
}

// Daten vorbereiten
function prepareData(data) {
    return tf.tensor2d(data.xValues, [data.xValues.length, 1]);
}

// Modell trainieren und Loss berechnen
async function trainModel(model, trainData, testData, epochs, batchSize = 32) {
    const xTrain = prepareData(trainData);
    const yTrain = tf.tensor2d(trainData.yValues, [trainData.yValues.length, 1]);

    const xTest = prepareData(testData);
    const yTest = tf.tensor2d(testData.yValues, [testData.yValues.length, 1]);

    const history = {
        epochs: [],
        trainLoss: [],
        testLoss: []
    };

    await model.fit(xTrain, yTrain, {
        epochs,
        batchSize,
        validationData: [xTest, yTest],
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                history.epochs.push(epoch + 1);
                history.trainLoss.push(logs.loss);
                history.testLoss.push(logs.val_loss);
            }
        }
    });

    const trainLoss = model.evaluate(xTrain, yTrain).dataSync()[0];
    const testLoss = model.evaluate(xTest, yTest).dataSync()[0];

    console.log('History:', history);

    return history;
}

function plotFinalLossHistory(ctx, history) {
    if (ctx.chart) {
        ctx.chart.destroy();
    }

    ctx.chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: history.epochs,
            datasets: [
                {
                    label: 'Train Loss',
                    data: history.trainLoss,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    fill: true,
                },
                {
                    label: 'Test Loss',
                    data: history.testLoss,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    fill: true,
                }
            ],
        },
        options: {
            scales: {
                x: { title: { display: true, text: 'Epoch' } },
                y: { beginAtZero: true, title: { display: true, text: 'Loss' } },
            },
        },
    });
}

// Vorhersage plotten
function plotPrediction(ctx, model, data, color, title, lossElementId) {
    const xValues = Array.from({ length: 100 }, (_, i) => -2 + (i * 4) / 99);
    const yPredicted = model.predict(tf.tensor2d(xValues, [xValues.length, 1])).dataSync();

    if (ctx.chart) {
        ctx.chart.destroy();
    }

    ctx.chart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: `${title} - Prediction`,
                    data: xValues.map((x, i) => ({ x: x, y: yPredicted[i] })),
                    backgroundColor: 'rgba(210, 210, 210, 0.5)',
                    showLine: true,
                    fill: false,
                },
                {
                    label: `${title}`,
                    data: data.xValues.map((x, i) => ({ x: x, y: data.yValues[i] })),
                    backgroundColor: color,
                },
            ],
        },
        options: {
            scales: {
                x: { type: 'linear', position: 'bottom' },
                y: { beginAtZero: true },
            },
        },
    });

    const lossElement = document.getElementById(lossElementId);
    const lossValue = model.evaluate(prepareData(data), tf.tensor2d(data.yValues, [data.yValues.length, 1])).dataSync()[0].toFixed(4);
    lossElement.textContent = `${lossValue}`;
}

// Hauptfunktion
async function main() {
    showLoader();

    const modelClean = createModel();
    const cleanHistory = await trainModel(modelClean, trainClean, testClean, 150);
    plotPrediction(ctxUnrauschedPredictionTrain, modelClean, trainClean, trainColor, 'Vorhersage ohne Rauschen (Train)', 'lossUnrauschedTrain');
    plotPrediction(ctxUnrauschedPredictionTest, modelClean, testClean, testColor, 'Vorhersage ohne Rauschen (Test)', 'lossUnrauschedTest');
    plotFinalLossHistory(document.getElementById('chartUnrauschedLossHistory').getContext('2d'), cleanHistory);

    const modelBestFit = createModel();
    const bestFitHistory = await trainModel(modelBestFit, trainNoisy, testNoisy, 85);
    plotPrediction(ctxBestFitPredictionTrain, modelBestFit, trainNoisy, trainColor, 'Vorhersage mit Best-Fit Modell (Train)', 'lossBestFitTrain');
    plotPrediction(ctxBestFitPredictionTest, modelBestFit, testNoisy, testColor, 'Vorhersage mit Best-Fit Modell (Test)', 'lossBestFitTest');
    plotFinalLossHistory(document.getElementById('chartBestFitLossHistory').getContext('2d'), bestFitHistory);

    const modelOverfit = createModel();
    const overfitHistory = await trainModel(modelOverfit, trainNoisy, testNoisy, 400);
    plotPrediction(ctxOverfitPredictionTrain, modelOverfit, trainNoisy, trainColor, 'Vorhersage mit Overfit-Modell (Train)', 'lossOverfitTrain');
    plotPrediction(ctxOverfitPredictionTest, modelOverfit, testNoisy, testColor, 'Vorhersage mit Overfit-Modell (Test)', 'lossOverfitTest');
    plotFinalLossHistory(document.getElementById('chartOverfitLossHistory').getContext('2d'), overfitHistory);

    hideLoader();
}

// Starte die Hauptfunktion
main();
