const { exec } = require('child_process');

// Replace 'python' with 'python3' if needed
const pythonScript = 'python predictor.py';

exec(pythonScript, (error, stdout, stderr) => {
  if (error) {
    console.error(`Error: ${error.message}`);
    return;
  }

  if (stderr) {
    console.error(`Error: ${stderr}`);
    return;
  }

  console.log(`Output: ${stdout}`);
});
