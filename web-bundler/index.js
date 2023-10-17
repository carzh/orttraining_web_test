// const ort = require('onnxruntime-web');
const ort = require('onnxruntime-web/training');

const chkptPath = '/../assets/artifacts/mnist/checkpoint.ckpt';
const trainingPath = '/../assets/artifacts/mnist/training_model.onnx';
const optimizerPath = '/../assets/artifacts/mnist/optimizer_model.onnx';
const evalPath = '/../assets/artifacts/mnist/eval_model.onnx';
// const inferenceModelPath = './assets/artifacts/inference/model.onnx';

// ort.env.wasm.numThreads = 2;
// ort.env.wasm.proxy = true;
// const chkptPath = '/../assets/artifacts/mobilevit/checkpoint.ckpt';
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

const onlyTrainCheckpointOptions = {
			checkpointState: chkptPath, 
			trainModel: trainingPath, 
			};

async function main() {
	try {
		const targets = await filePathToTensorInt(targetPath);
		const data = await filePathToTensorFloat(dataPath);
		console.log('after loading the file and attempting to write');
		// const is = await ort.InferenceSession.create(inferenceModelPath);
		// const is2 = await ort.InferenceSession.create(inferenceModelPath);

		// console.log("after loading inference sessions");
		// document.write("successfully loaded 2 inference sessions");
		
		const ts = await ort.TrainingSession.create(onlyTrainCheckpointOptions);
		console.log('the ts inputNames is', ts.inputNames);

		console.log('before loading file');
		document.write('loading file');

		const feeds = { "input": data, "labels": targets };

		const results = await ts.runTrainStep(feeds);
		document.write('<br/>');
		for (const key in results) {
			document.write(key);
			document.write(results[key].data);
			document.write('<br/>');
		}
		
		document.write('loading success!!!! whoohooooo~~');
	} catch (e) {
		document.write('<br/>:(<br/>');
		document.write(`FAILURE: ${e}.`);
		document.write('<br/>:(<br/>');
		document.write(e.stack);
	}
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

main();
