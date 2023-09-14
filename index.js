const ort = require('onnxruntime-web/training');

const chkptPath = 'assets/artifacts/mobilevit/checkpoint.ckpt';
const trainingPath = 'assets/artifacts/mobilevit/training_model.onnx';
const optimizerPath = 'assets/artifacts/mobilevit/optimizer_model.onnx';
const evalPath = 'assets/artifacts/mobilevit/eval_model.onnx';

const dataPath = 'assets/data/train-images.txt';
const targetPath = 'assets/data/train-labels.txt';

async function main() {
	try {
		// const ts = await ort.TrainingSession.create(chkptPath, trainingPath, evalPath, optimizerPath);

		console.log('before loading file');
		document.write('loading file');

		const targets = await filePathToTensor(targetPath);
		document.write(targets);
		console.log('after loading the file and attempting to write');
		document.write('<br/>');
		document.write('loading success!!!! whoohooooo~~');
	} catch (e) {
		console.log('in the caatch block');
		document.write('<br/>');
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
      if (readerDone) {
		console.log("in the readerDone if statement");
		break;
	  }
	  console.log('reading the last chunk');
      const remainder = chunk.substr(startIndex);
      ({ value: chunk, done: readerDone } = await reader.read());
      chunk = remainder + (chunk ? utf8Decoder.decode(chunk) : "");
      startIndex = newline.lastIndex = 0;
      continue;
    }
    yield chunk.substring(startIndex, result.index);
    startIndex = newline.lastIndex;
  }
  console.log('outside of the while loop');

  if (startIndex < chunk.length) {
    // Last line didn't end in a newline char
    yield chunk.substr(startIndex);
  }
}

/**
 * written to only process up to 2d arrays
 * @param {}} filePath 
 */
async function filePathToTensor(filePath) {
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
  console.log('after for loop');
  let dims = [dim_0, dim_1];
  document.write(list);
  const fixedFloatArray = new Float32Array(list);
  
  return new ort.Tensor('float32', fixedFloatArray, dims);
}

main();
