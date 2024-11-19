const { app, BrowserWindow } = require('electron');
const { exec } = require('child_process');
const net = require('net');

let mainWindow;

const DASH_PORT = 8050; // Dash app port
const DASH_URL = `http://127.0.0.1:${DASH_PORT}`;
const DASH_SCRIPT = 'python app-work.py'; // Command to run your Dash app

// Function to check if the Dash server is running
function checkServer(port, callback) {
    const client = new net.Socket();

    client.connect({ port }, () => {
        client.end();
        callback(true);
    });

    client.on('error', () => {
        callback(false);
    });
}

// Wait for the Dash server to start
function waitForServer(port, retries = 20, interval = 500) {
    return new Promise((resolve, reject) => {
        const attemptConnection = (remainingRetries) => {
            if (remainingRetries === 0) {
                return reject(new Error('Dash server not started.'));
            }

            checkServer(port, (isRunning) => {
                if (isRunning) {
                    return resolve();
                }

                setTimeout(() => attemptConnection(remainingRetries - 1), interval);
            });
        };

        attemptConnection(retries);
    });
}

// Start the Dash app and Electron app
app.on('ready', async () => {
    console.log('Starting Dash app...');
    const dashProcess = exec(DASH_SCRIPT);

    dashProcess.stdout.on('data', (data) => console.log(`Dash: ${data}`));
    dashProcess.stderr.on('data', (data) => console.error(`Dash Error: ${data}`));

    try {
        await waitForServer(DASH_PORT);
        console.log('Dash server is running. Launching Electron app...');

        mainWindow = new BrowserWindow({
            width: 1280,
            height: 720,
            webPreferences: {
                contextIsolation: true,
            },
        });

        mainWindow.loadURL(DASH_URL);

        mainWindow.on('closed', () => {
            mainWindow = null;
            dashProcess.kill();
        });
    } catch (error) {
        console.error(error.message);
        app.quit();
        dashProcess.kill();
    }
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});
