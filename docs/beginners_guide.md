### 1. Initialize the Cloud Resources
Before the website can work, you must wake up the "Brain" on AWS.
* Open the file `SageWall_Training.ipynb` in your editor.
* Select "Run All Cells" to execute the notebook.
* **Wait:** This process takes about 3 to 5 minutes. It is provisioning a dedicated server on AWS.
* **Success:** Look for the output line at the bottom that says: `Endpoint Name: sagemaker-xgboost-YYYY-MM...`

### 2. Configure the Dashboard
Once the endpoint is live, you need to connect this website to it.
* Copy the **Endpoint Name** from the notebook output.
* Paste it into the **SageMaker Endpoint Name** field in the sidebar on the left.
* **Note:** If the name is incorrect, the app will show a connection error.

### 3. Simulate an Attack
You do not need to be a hacker to test this. We can inject a "Feature Vector" that mimics an attack.
* Locate the **Packet Data Input** box on the main screen.
* Paste a comma separated string of 41 numbers.
* **Tip:** You can use the default value provided, or copy a row from the `data/sample_attack.csv` file.

### 4. Analyze the Result
Click the **Scan Packet** button.
* **Confidence Score:** This number represents the AI's certainty. A score above 0.50 usually indicates a threat.
* **Action:** If the score is high, the system simulates blocking the IP. If low, it allows the traffic to pass.

### 5. Important: Cost Management
When you are finished testing, you **must** shut down the AWS resources to avoid billing.
* Return to the `SageWall_Training.ipynb` notebook.
* Run the final cell labeled **Delete Endpoint**.
* Verify in the AWS Console that the endpoint status is "Deleted."