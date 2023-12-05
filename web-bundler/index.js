// const ort = require('onnxruntime-web');
const ort = require('onnxruntime-web/training');

const chkptPath = '/../assets/artifacts/mnist/checkpoint.ckpt';
const trainingPath = '/../assets/artifacts/mnist/training_model.onnx';
const optimizerPath = '/../assets/artifacts/mnist/optimizer_model.onnx';
const evalPath = '/../assets/artifacts/mnist/eval_model.onnx';
const inferenceModelPath = './assets/artifacts/inference/model.onnx';

const chkptPathTrainingApi = '/../assets/artifacts/trainingapi/checkpoint.ckpt';
const trainingPathTrainingApi  = '/../assets/artifacts/trainingapi/training_model.onnx';
const optimizerPathTrainingApi  = '/../assets/artifacts/trainingapi/adamw.onnx';
const evalPathTrainingApi  = '/../assets/artifacts/trainingapi/eval_model.onnx';
const inferenceModelPathTrainingApi  = './assets/artifacts/inference/model.onnx';

// ort.env.wasm.numThreads = 2;
// ort.env.wasm.numThreads = 2;
// ort.env.wasm.proxy = true;
// const chkptPathMobilevit = '/../assets/artifacts/mobilevit/checkpoint.ckpt';
// const trainingPath = '/../assets/artifacts/mobilevit/training_model.onnx';
// const optimizerPath = '/../assets/artifacts/mobilevit/optimizer_model.onnx';
// const evalPath = '/../assets/artifacts/mobilevit/eval_model.onnx';

const dataPath = 'assets/data/train-images.txt';
const targetPath = 'assets/data/train-labels.txt';

const allOptions = {
			checkpointState: chkptPath, 
			trainModel: trainingPath, 
			evalModel: evalPath, 
			optimizerModel: optimizerPath};

const allOptionsTrainingApi = {
			checkpointState: chkptPathTrainingApi , 
			trainModel: trainingPathTrainingApi , 
			evalModel: evalPathTrainingApi , 
			optimizerModel: optimizerPathTrainingApi };

const onlyTrainCheckpointOptions = {
			checkpointState: chkptPath, 
			trainModel: trainingPath, 
			};

async function main() {
	try {
		const targets = await filePathToTensorInt(targetPath);
		const data = await filePathToTensorFloat(dataPath);
		console.log('after loading the file and attempting to write');
		const is = await ort.InferenceSession.create(inferenceModelPath);
		const is2 = await ort.InferenceSession.create(inferenceModelPath);

		console.log("after loading inference sessions");
		document.write("successfully loaded 2 inference sessions");
		is.release();
		
		const ts = await ort.TrainingSession.create(allOptions);
		console.log('the ts inputNames is', ts.inputNames);

		console.log('before loading file');
		document.write('loading file');
		document.write(data.dims);

		let feeds = { "input": data, "labels": targets };

		await ts.lazyResetGrad();

		await runTrainStepAndWriteResults(ts, feeds);

		await runEvalStepAndWriteResults(ts, feeds);

		await writeContiguousParameters(ts);
		await ts.runOptimizerStep(feeds);
		document.write("OPTIMIZER STEP HAPPENED;");

		document.write('<br/>');
		await writeContiguousParameters(ts);
		feeds = { "input": data, "labels": targets };
		await runEvalStepAndWriteResults(ts, feeds);

		await ts.lazyResetGrad();
		// feeds = { "input": data, "labels": targets };
		await runTrainStepAndWriteResults(ts, feeds);
		
		const paramsLength = await ts.getParametersSize();
		document.write('<br/>');
		document.write('<br/>');
		document.write('parameters length: ');
		document.write(paramsLength);

		const newParamVal = -1.5;
		const testFloatOne = await createConstantFloat32Array(paramsLength, newParamVal);
		document.write('<br/>');
		document.write('float 32 array created');
		document.write(testFloatOne);
		document.write('<br/>');
		const testUintOne = new Uint8Array(testFloatOne.buffer, testFloatOne.byteOffset, testFloatOne.byteLength);
		document.write('<br/>');
		document.write('uint 8 array created');
		document.write(testUintOne);
		document.write('<br/>');

		await ts.loadParametersBuffer(testUintOne);

		document.write('<br/>');
		document.write(`trainable params after load attempt -- should be all ${newParamVal}s`);
		await writeContiguousParameters(ts);

		await ts.release();

		document.write('<br/>');
		document.write('<br/>');
		document.write('success!!!! whoohooooo~~');

		document.write('<br/>');
		document.write('<br/>');
		document.write('TRAINING API TEST ====================================================================');
		document.write('<br/>');
		const tsTrainingApi = await ort.TrainingSession.create(allOptionsTrainingApi);
		const input0 = new ort.Tensor('float32', generateGaussianFloatArray(2 * 784), [2, 784]);
		const labels = new ort.Tensor('int32', [2, 1], [2]);
		const feedsTrainingApi = {"input-0": input0, "labels": labels};

		await runTrainStepAndWriteResults(tsTrainingApi, feedsTrainingApi);

	} catch (e) {
		document.write('<br/>:(<br/>');
		document.write(`FAILURE: ${e}.`);
		document.write('<br/>:( here is the call stack:<br/>');
		document.write(e.stack);
	}
}

async function writeResults(results, funcName) {
		document.write('<br/>');
		document.write('run ' + funcName + ' results');
		document.write('<br/>');
		for (const key in results) {
			document.write(key);
			document.write(": ");
			document.write(results[key].data);
			document.write('<br/>');
		}
}

async function runTrainStepAndWriteResults(ts, feeds) {
		const results = await ts.runTrainStep(feeds);
		writeResults(results, 'trainStep');
}

async function runEvalStepAndWriteResults(ts, feeds) {
		const results = await ts.runEvalStep(feeds);
		writeResults(results, 'evalStep');
}

async function writeContiguousParameters(ts) {
		const trainableParams = await ts.getContiguousParameters();
		document.write('<br/>');
		document.write('writing contiguous parameters');
		document.write('<br/>');
		document.write(trainableParams.data);
		document.write('<br/>');
		document.write('is float32:');
		document.write('<br/>');
		document.write(trainableParams.data.constructor === Float32Array);
		document.write('<br/>');
		document.write('<br/>');
}

async function* makeTextFileLineIterator(fileURL) {
  const utf8Decoder = new TextDecoder("utf-8");
  const response = await fetch(fileURL);
  const reader = response.body.getReader();
  let { value: chunk, done: readerDone } = await reader.read();
  chunk = chunk ? utf8Decoder.decode(chunk) : "";

  const newline = /\r?\n/gm;
  let startIndex = 0;
  let result;

  while (true) {
    const result = newline.exec(chunk);
    if (!result) {
      if (readerDone) break;
      const remainder = chunk.substr(startIndex);
      ({ value: chunk, done: readerDone } = await reader.read());
      chunk = remainder + (chunk ? utf8Decoder.decode(chunk) : "");
      startIndex = newline.lastIndex = 0;
      continue;
    }
    yield chunk.substring(startIndex, result.index);
    startIndex = newline.lastIndex;
  }

  if (startIndex < chunk.length) {
    // Last line didn't end in a newline char
    yield chunk.substr(startIndex);
  }
}

/**
 * written to only process up to 2d arrays
 * @param {}} filePath 
 */
async function filePathToTensorFloat(filePath, dims) {
	let list = new Array()
	let dim_0 = 0;
	let dim_1 = 0;
  for await (const line of makeTextFileLineIterator(filePath)) {
	if (line.length == 0) {
		continue;
	}
	let words = line.split(" ");
	for (const word of words) {
		list.push(parseFloat(word));
	}
	dim_1 = words.length;
	dim_0 += 1;
  }
  if (typeof dims === 'undefined') {
  	dims = [dim_0, dim_1];
  }
  const fixedFloatArray = new Float32Array(list);
  
  return new ort.Tensor('float32', fixedFloatArray, dims);
}

/**
 * written to only process up to 1d arrays
 * @param {}} filePath 
 */
async function filePathToTensorInt(filePath, dims) {
	let list = new Array()
	let dim_0 = 0;
  for await (const line of makeTextFileLineIterator(filePath)) {
	if (line.length == 0) {
		continue;
	}
	let words = line.split(" ");
	for (const word of words) {
		if (word == "NaN") {
			console.log("continuing after nan found ", word, line);
			continue;
		}
		list.push(BigInt(parseInt(word)));
	}
	dim_0 += 1;
  }
  if (typeof dims === 'undefined') {
  	dims = [dim_0];
  }
  const fixedFloatArray = new BigInt64Array(list);
  
  return new ort.Tensor('int64', fixedFloatArray, dims);
}

// makes a float32array of the specified length, populated only with the given constant
function createConstantFloat32Array(length, constant) {
	const arr = new Float32Array(length);

	for (i = 0; i < length; i++) {
		arr[i] = constant;
	}
	return arr;
};

function generateGaussianRandom(mean=0, scale=1) {
  const u = 1 - Math.random();
  const v = Math.random();
  const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  return z * scale + mean;
}

function generateGaussianFloatArray(length) {
  const array = new Float32Array(length);

  for (let i = 0; i < length; i++) {
    array[i] = generateGaussianRandom();
  }

  return array;
}


main();
