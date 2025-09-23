// Test script to verify that the SAAFE dashboard is working correctly

const http = require('http');

// Test the frontend
function testFrontend() {
  return new Promise((resolve, reject) => {
    const req = http.get('http://localhost:5173', (res) => {
      if (res.statusCode === 200) {
        resolve('Frontend is running correctly');
      } else {
        reject(new Error(`Frontend returned status code: ${res.statusCode}`));
      }
    });
    
    req.on('error', (err) => {
      reject(new Error(`Failed to connect to frontend: ${err.message}`));
    });
    
    req.setTimeout(5000, () => {
      req.destroy();
      reject(new Error('Frontend connection timed out'));
    });
  });
}

// Test the backend API
function testBackend() {
  return new Promise((resolve, reject) => {
    const req = http.get('http://localhost:8000/api/fire-detection-data', (res) => {
      let data = '';
      
      res.on('data', (chunk) => {
        data += chunk;
      });
      
      res.on('end', () => {
        if (res.statusCode === 200) {
          try {
            const jsonData = JSON.parse(data);
            if (jsonData.status === 'success') {
              resolve('Backend API is running correctly');
            } else {
              reject(new Error('Backend API returned unexpected response'));
            }
          } catch (err) {
            reject(new Error('Backend API returned invalid JSON'));
          }
        } else {
          reject(new Error(`Backend API returned status code: ${res.statusCode}`));
        }
      });
    });
    
    req.on('error', (err) => {
      reject(new Error(`Failed to connect to backend: ${err.message}`));
    });
    
    req.setTimeout(5000, () => {
      req.destroy();
      reject(new Error('Backend connection timed out'));
    });
  });
}

// Run tests
async function runTests() {
  console.log('Testing SAAFE Fire Detection Dashboard...\n');
  
  try {
    const frontendResult = await testFrontend();
    console.log(`✓ ${frontendResult}`);
  } catch (err) {
    console.log(`✗ Frontend test failed: ${err.message}`);
  }
  
  try {
    const backendResult = await testBackend();
    console.log(`✓ ${backendResult}`);
  } catch (err) {
    console.log(`✗ Backend test failed: ${err.message}`);
  }
  
  console.log('\nTest completed.');
}

runTests();