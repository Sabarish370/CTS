# AWS S3 Configuration for CTS Hackathon Pipeline

## Overview
The pipeline has been updated to support AWS S3 for storing raw data and analytical results. This eliminates the need to keep generated data in the project folder, making it suitable for deployment on AWS EC2.

## How S3 Mode Works

When `S3_ENABLED=true`, the pipeline uses **intelligent path redirection** via symlinks:

1. **Data Download**: Raw data from S3 is downloaded to `/tmp/cts-pipeline/generated_data`
2. **Symlink Creation**: Before running any scripts, symlinks are created:
   ```
   project_root/generated_data       → /tmp/cts-pipeline/generated_data
   project_root/preprocessed_data    → /tmp/cts-pipeline/preprocessed_data
   project_root/matched_pairs        → /tmp/cts-pipeline/matched_pairs
   project_root/did_roi_output       → /tmp/cts-pipeline/did_roi_output
   ```
3. **Script Execution**: All subprocess scripts find their data through symlinks transparently
4. **Result Upload**: Analytical results uploaded from `/tmp/cts-pipeline/` to S3
5. **Cleanup**: Symlinks removed after pipeline completes

This approach ensures **zero code changes** to subprocess scripts—they continue using `PROJECT_ROOT` paths without modification.

## Architecture

```
┌─────────────────────────────────────────┐
│  AWS S3 Bucket                          │
├─────────────────────────────────────────┤
│  raw-data/                              │  (Input: generated_data files)
│  ├── generated_data/                    │
│  ├── attended.csv                       │
│  ├── events.csv                         │
│  ├── hcp.csv                            │
│  ├── rx_claims_monthly.csv              │
│  └── ground_truth_config.json           │
│                                         │
│  analytical-data/                       │  (Output: Results)
│  ├── matched_pairs/                     │
│  │   ├── nearest_neigbour_matching/     │
│  │   ├── rule_based_matching/           │
│  │   ├── propensity_score_matching/     │
│  │   └── randam_matching/               │
│  └── did_roi_output/                    │
│      ├── did_roi_results_*.csv          │
│      └── did_roi_summary_*.csv          │
└─────────────────────────────────────────┘
```

## Windows Compatibility (Important)

On Windows, symlinks require **special permissions**. Choose ONE approach:

### Option 1: Developer Mode (Windows 10/11 - Recommended)
1. Open **Settings** → **Privacy & Security** → **For developers**
2. Toggle ON: **"Developer Mode"**
3. Click **"Yes"** to confirm

This allows symlinks without admin privileges. After enabling, restart terminal if needed.

### Option 2: Admin Command Prompt
Run command prompt as Administrator and create symlinks manually:
```cmd
cd C:\path\to\project
mklink /D generated_data \tmp\cts-pipeline\generated_data
mklink /D preprocessed_data \tmp\cts-pipeline\preprocessed_data
mklink /D matched_pairs \tmp\cts-pipeline\matched_pairs
mklink /D did_roi_output \tmp\cts-pipeline\did_roi_output
```

### Option 3: Use WSL2 (Windows Subsystem for Linux)
```bash
# In WSL2 terminal
cd /mnt/d/CTS/CTS_Hackathon_git
export S3_ENABLED=true
export S3_BUCKET=your-bucket-name
python pipeline.py --stage all
```

WSL2 has full symlink support without special configuration.

### Option 4: Alternative - No Symlinks (Fallback)
If symlinks fail on Windows and you cannot enable Developer Mode:
- Download analytical results from S3 after pipeline completes
- Copy manually to project directory for dashboard viewing

## Prerequisites

1. **AWS Account & S3 Bucket**: Create an S3 bucket to store data
   ```bash
   aws s3 mb s3://your-bucket-name --region us-east-1
   ```

2. **AWS Credentials**: Configure AWS credentials on EC2
   ```bash
   # Option 1: IAM Role (Recommended for EC2)
   # Attach IAM role with S3 access to EC2 instance
   
   # Option 2: AWS CLI Configuration
   aws configure
   # Enter: Access Key ID, Secret Access Key, Default region
   ```

3. **Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install boto3
   ```

4. **Symlink Support** (Windows only)
   - See Windows Compatibility section above
   - Linux/macOS have symlinks enabled by default
   - EC2 instances (Amazon Linux/Ubuntu) have symlinks enabled by default

## Environment Variables

Set these environment variables before running the pipeline:

```bash
# Enable S3 mode
export S3_ENABLED=true

# S3 Bucket name (required if S3_ENABLED=true)
export S3_BUCKET=your-bucket-name

# AWS Region (default: us-east-1)
export S3_REGION=us-east-1
```

### Example: Setting Environment Variables on EC2

```bash
# In ~/.bashrc or before running pipeline
export S3_ENABLED=true
export S3_BUCKET=cts-hackathon-data
export S3_REGION=us-east-1

# Verify settings
echo $S3_ENABLED $S3_BUCKET $S3_REGION
```

## Data Upload to S3

Before running the pipeline for the first time, upload your raw data to S3:

```bash
# Upload generated_data to raw-data prefix
aws s3 sync ./generated_data s3://your-bucket-name/raw-data/generated_data

# Or upload individual files
aws s3 cp generated_data/attendance.csv s3://your-bucket-name/raw-data/
aws s3 cp generated_data/events.csv s3://your-bucket-name/raw-data/
aws s3 cp generated_data/hcp.csv s3://your-bucket-name/raw-data/
aws s3 cp generated_data/rx_claims_monthly.csv s3://your-bucket-name/raw-data/
aws s3 cp generated_data/ground_truth_config.json s3://your-bucket-name/raw-data/
```

## Running the Pipeline with S3

### Full Pipeline with S3
```bash
export S3_ENABLED=true
export S3_BUCKET=cts-hackathon-data
python pipeline.py --stage all
```

The pipeline will:
1. **Create symlinks** pointing from project directories to `/tmp/cts-pipeline/`
2. **Download raw data** from `s3://bucket/raw-data/` to `/tmp/cts-pipeline/generated_data`
3. **Run preprocessing** (reads from symlinked paths)
4. **Run 4 matching methods** (reads from symlinked paths)
5. **Run DiD/ROI analysis** (reads from symlinked paths)
6. **Upload results** to `s3://bucket/analytical-data/`
7. **Clean up symlinks** automatically after completion

### Specific Stages
```bash
# Preprocessing only
python pipeline.py --stage preprocess

# Matching only (requires preprocessed data)
python pipeline.py --stage match --methods nnm psm

# DiD/ROI analysis only
python pipeline.py --stage did_roi

# Validate schemas
python pipeline.py --stage validate
```

## What Happens During Pipeline Execution

### 1. **Download Phase** (Preprocess stage)
   - Pipeline downloads all files from `s3://bucket/raw-data/` 
   - Files stored in temporary directory: `/tmp/cts-pipeline/`
   - Downloads: attendance.csv, events.csv, hcp.csv, rx_claims_monthly.csv, ground_truth_config.json

### 2. **Processing Phase** (Preprocess → Match → DiD/ROI)
   - All intermediate files stored in `/tmp/cts-pipeline/`
   - Preprocessing: Creates `preprocessed_data/`
   - Matching: Creates `matched_pairs/` with method-specific subdirectories
   - DiD/ROI: Creates `did_roi_output/` with results and summary files

### 3. **Upload Phase** (After DiD/ROI completes)
   - Upload `matched_pairs/` → `s3://bucket/analytical-data/matched_pairs/`
   - Upload `did_roi_output/` → `s3://bucket/analytical-data/did_roi_output/`
   - Local temporary files in `/tmp/cts-pipeline/` are NOT automatically cleaned

## Local Development (S3 Disabled)

To run locally without S3 (testing/development):

```bash
# Unset S3 environment variables or set to false
export S3_ENABLED=false

# Run normally - uses ./preprocessed_data, ./matched_pairs, ./did_roi_output
python pipeline.py --stage all
```

## Directory Structure

### With S3 Enabled
```
project/
├── pipeline.py                    (Modified for S3)
├── Preprocessing_tasks/
├── matching_techinques/
├── did_roi_engine.py
├── dashboard.py
├── requirements.txt
├── pipeline_logs/                 (Logs stay local)
└── S3_SETUP.md                    (This file)

/tmp/cts-pipeline/                 (Temporary S3 staging)
├── generated_data/                (Downloaded from raw-data/)
├── preprocessed_data/             (Generated, then uploaded)
├── matched_pairs/                 (Generated, then uploaded)
└── did_roi_output/                (Generated, then uploaded)
```

### With S3 Disabled
```
project/
├── generated_data/                (Local)
├── preprocessed_data/             (Local)
├── matched_pairs/                 (Local)
├── did_roi_output/                (Local)
└── pipeline_logs/
```

## Troubleshooting

### Error: "boto3 is not installed"
```bash
pip install boto3
```

### Error: "Failed to create symlink"
**Windows Users**: See Windows Compatibility section above
- Enable Developer Mode, or
- Run as Administrator, or
- Use WSL2, or
- Run with S3_ENABLED=false and manually manage directories

**Linux/macOS Users**: Check permissions:
```bash
ls -la ~/tmp/cts-pipeline  # Verify directory is readable/writable
```

### Error: "S3 download failed: NoSuchBucket"
- Verify bucket name in S3_BUCKET environment variable
- Check AWS credentials: `aws s3 ls`
- Ensure bucket exists: `aws s3 ls s3://your-bucket-name`

### Error: "Permission denied" for S3 operations
- Verify IAM role has S3 permissions on EC2
- Check credentials: `aws sts get-caller-identity`
- Ensure bucket policy allows the IAM user/role

### Error: "FileNotFoundError: generated_data not found"
- **On Windows**: Symlink creation failed (see Windows Compatibility section)
- **On Linux/macOS/EC2**: Check `/tmp/cts-pipeline/generated_data` exists:
  ```bash
  ls -la /tmp/cts-pipeline/generated_data
  # If missing, ensure raw data was uploaded to S3:
  aws s3 ls s3://your-bucket/raw-data/ --recursive
  ```

### Symlinks Not Working After Pipeline Completes
Symlinks are automatically cleaned up after the pipeline finishes. This is normal.
- If dashboard needs results, ensure data was uploaded to S3
- Download results manually: `aws s3 sync s3://bucket/analytical-data ./local-results`

### Data Not Found in S3
```bash
# List files in raw-data prefix
aws s3 ls s3://your-bucket-name/raw-data/

# List files in analytical-data prefix
aws s3 ls s3://your-bucket-name/analytical-data/ --recursive
```

### Cleaning Up Temporary Files
```bash
# Remove temporary pipeline staging directory
rm -rf /tmp/cts-pipeline
```

## Security Best Practices

1. **Use IAM Roles** (EC2): Attach IAM role instead of using access keys
2. **Bucket Versioning**: Enable S3 versioning for data recovery
   ```bash
   aws s3api put-bucket-versioning --bucket your-bucket-name \
     --versioning-configuration Status=Enabled
   ```
3. **Encryption**: Enable S3 bucket encryption
   ```bash
   aws s3api put-bucket-encryption --bucket your-bucket-name \
     --server-side-encryption-configuration '{...}'
   ```
4. **Access Logging**: Enable S3 access logs for audit trails
5. **Lifecycle Policies**: Archive old analytical results after retention period

## Cost Optimization

- Use S3 Standard storage for active data
- Archive analytical results to S3 Glacier after 30 days
- Use S3 Transfer Acceleration for faster uploads from EC2
- Monitor usage: `aws s3 ls s3://your-bucket-name --summarize --human-readable --recursive`

## Dashboard with S3

After pipeline completes, download analytical results for dashboard:

```bash
# Download results for local dashboard viewing
aws s3 sync s3://your-bucket-name/analytical-data ./local-results

# Or run dashboard against S3 directly (requires dashboard.py modifications)
```

## Integration with CI/CD

Example GitHub Actions workflow:

```yaml
name: CTS Pipeline with S3
on: [push]
jobs:
  pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v1
        with:
          role-to-assume: arn:aws:iam::ACCOUNT:role/github-actions
          aws-region: us-east-1
      - name: Run pipeline
        env:
          S3_ENABLED: 'true'
          S3_BUCKET: ${{ secrets.S3_BUCKET }}
        run: python pipeline.py --stage all
```

## Additional Resources

- [AWS S3 Documentation](https://docs.aws.amazon.com/s3/)
- [boto3 S3 Client](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html)
- [AWS CLI S3 Commands](https://docs.aws.amazon.com/cli/latest/reference/s3/)
