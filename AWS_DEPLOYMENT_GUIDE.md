# AWS EC2 Deployment Guide - Step by Step

Complete guide to deploy CTS Hackathon pipeline on AWS EC2 with S3 storage.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [AWS Account Setup](#aws-account-setup)
3. [S3 Bucket Creation](#s3-bucket-creation)
4. [IAM Role Setup](#iam-role-setup)
5. [EC2 Instance Launch](#ec2-instance-launch)
6. [Instance Configuration](#instance-configuration)
7. [Project Deployment](#project-deployment)
8. [Upload Raw Data](#upload-raw-data)
9. [Run the Pipeline](#run-the-pipeline)
10. [View Results](#view-results)
11. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before starting, ensure you have:
- ✅ AWS account with billing enabled
- ✅ AWS CLI installed on your local machine ([Download](https://aws.amazon.com/cli/))
- ✅ AWS credentials configured locally: `aws configure`
- ✅ SSH key pair (or willing to create one)
- ✅ Project repository cloned: `https://github.com/Sabarish370/CTS.git`

**Verify AWS CLI:**
```bash
aws --version
aws sts get-caller-identity
```

---

## AWS Account Setup

### Step 1: Sign in to AWS Console

1. Go to [AWS Management Console](https://console.aws.amazon.com)
2. Sign in with your AWS account
3. Verify you're in the correct region (default: **us-east-1**)

**Check Region:**
- Look at top-right corner of AWS Console
- Should show "us-east-1" or your preferred region
- Click to change if needed

---

## S3 Bucket Creation

### Step 2: Create S3 Bucket for Data Storage

**Via AWS Console:**

1. Open **S3** service from AWS Console
2. Click **"Create bucket"**
3. **Bucket name**: `cts-hackathon-data` (or your preferred name)
   - Must be globally unique
   - Only lowercase letters, numbers, hyphens
4. **Region**: `us-east-1` (or your preferred region)
5. **Block Public Access settings**: Keep ALL blocked (default)
6. Click **"Create bucket"**

**Via AWS CLI (faster):**
```bash
aws s3 mb s3://cts-hackathon-data --region us-east-1
```

**Verify bucket created:**
```bash
aws s3 ls
# Should show: 2026-08-18 15:30:45 cts-hackathon-data
```

### Step 3: Create S3 Folder Structure

Create two folders in the bucket:

```bash
# Create raw-data folder
aws s3 mb s3://cts-hackathon-data/raw-data/ --region us-east-1

# Create analytical-data folder  
aws s3 mb s3://cts-hackathon-data/analytical-data/ --region us-east-1
```

**Verify structure:**
```bash
aws s3 ls s3://cts-hackathon-data/ --recursive
```

---

## IAM Role Setup

### Step 4: Create IAM Role for EC2

**Via AWS Console:**

1. Open **IAM** service
2. Click **"Roles"** in left sidebar
3. Click **"Create role"**
4. **Trusted entity type**: Select "AWS service"
5. **Use case**: Search for and select **"EC2"**
6. Click **"Next"**
7. **Add permissions**: Search for **"AmazonS3FullAccess"**
   - Click checkbox next to "AmazonS3FullAccess"
8. Click **"Next"**
9. **Role name**: `cts-ec2-s3-role`
10. Click **"Create role"**

**Via AWS CLI (faster):**

Create trust policy file (`trust-policy.json`):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

Then create role:
```bash
# Create role
aws iam create-role \
  --role-name cts-ec2-s3-role \
  --assume-role-policy-document file://trust-policy.json

# Attach S3 full access policy
aws iam attach-role-policy \
  --role-name cts-ec2-s3-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Verify
aws iam get-role --role-name cts-ec2-s3-role
```

### Step 5: Create Instance Profile

**Via AWS CLI:**
```bash
# Create instance profile
aws iam create-instance-profile \
  --instance-profile-name cts-ec2-s3-profile

# Add role to instance profile
aws iam add-role-to-instance-profile \
  --instance-profile-name cts-ec2-s3-profile \
  --role-name cts-ec2-s3-role

# Verify
aws iam get-instance-profile \
  --instance-profile-name cts-ec2-s3-profile
```

---

## EC2 Instance Launch

### Step 6: Create SSH Key Pair

**Via AWS CLI:**
```bash
# Create and save key pair
aws ec2 create-key-pair \
  --key-name cts-deployment-key \
  --query 'KeyMaterial' \
  --output text > cts-deployment-key.pem

# Set permissions (IMPORTANT)
chmod 400 cts-deployment-key.pem

# Verify
ls -la cts-deployment-key.pem
```

**Save this file safely** - you'll need it to SSH into the instance.

### Step 7: Launch EC2 Instance

**Via AWS Console:**

1. Open **EC2** service
2. Click **"Instances"** → **"Launch instances"**
3. **Name**: `cts-pipeline-server`
4. **AMI**: Search for **"Amazon Linux 2"** → Select it
   - AMI ID should start with `ami-` and say "Amazon Linux 2"
5. **Instance type**: `t3.large` (2 vCPU, 8 GB RAM)
   - Good for processing pipeline
6. **Key pair**: Select **"cts-deployment-key"**
7. **Network settings**: Keep defaults (default VPC)
8. **Storage**: Keep default (8 GB gp2)
9. **Advanced details** → **IAM instance profile**: Select **"cts-ec2-s3-profile"**
10. Click **"Launch instance"**

**Via AWS CLI:**
```bash
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t3.large \
  --key-name cts-deployment-key \
  --iam-instance-profile Name=cts-ec2-s3-profile \
  --count 1 \
  --region us-east-1 \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=cts-pipeline-server}]'
```

### Step 8: Get Instance IP Address

**Via AWS Console:**
1. Go to **EC2** → **Instances**
2. Find **"cts-pipeline-server"**
3. Copy **"Public IPv4 address"** (e.g., `54.123.45.67`)

**Via AWS CLI:**
```bash
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=cts-pipeline-server" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text
```

**Wait 2-3 minutes for instance to be ready!**

---

## Instance Configuration

### Step 9: SSH into EC2 Instance

```bash
ssh -i cts-deployment-key.pem ec2-user@<PUBLIC_IP>
```

Replace `<PUBLIC_IP>` with the actual IP address (e.g., `54.123.45.67`)

**First login may take a minute. You should see:**
```
       __|  __|_  )
       _|  (     /   Amazon Linux 2
      ___|\___|___|

https://aws.amazon.com/amazon-linux-2/
[ec2-user@ip-172-31-XX-XX ~]$
```

### Step 10: Update System and Install Python

```bash
# Update system
sudo yum update -y

# Install Python 3 and pip
sudo yum install -y python3 python3-pip git

# Verify installations
python3 --version
pip3 --version
git --version
```

### Step 11: Clone Project Repository

```bash
# Clone the project
git clone https://github.com/Sabarish370/CTS.git ~/cts-pipeline
cd ~/cts-pipeline

# List files
ls -la
```

### Step 12: Install Python Dependencies

```bash
# Install all requirements
pip3 install -r requirements.txt

# Verify boto3 installed
python3 -c "import boto3; print('boto3 version:', boto3.__version__)"
```

### Step 13: Verify AWS Credentials (IAM Role)

```bash
# Test S3 access
aws s3 ls

# Test EC2 identity
aws sts get-caller-identity

# Should show:
# {
#     "UserId": "AIDAJ...",
#     "Account": "123456789012",
#     "Arn": "arn:aws:iam::123456789012:role/cts-ec2-s3-role"
# }
```

---

## Project Deployment

### Step 14: Configure Environment Variables

On the EC2 instance:

```bash
# Create .env file
cat > ~/cts-pipeline/.env << 'EOF'
S3_ENABLED=true
S3_BUCKET=cts-hackathon-data
S3_REGION=us-east-1
EOF

# Verify
cat ~/cts-pipeline/.env
```

### Step 15: Create Pipeline Execution Script

```bash
# Create execution script
cat > ~/cts-pipeline/run-pipeline.sh << 'EOF'
#!/bin/bash
set -e

cd ~/cts-pipeline

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

echo "=========================================="
echo "CTS Pipeline Execution on EC2"
echo "=========================================="
echo "S3 Mode: $S3_ENABLED"
echo "S3 Bucket: $S3_BUCKET"
echo "AWS Region: $S3_REGION"
echo "=========================================="
echo ""

# Run pipeline with all stages
python3 pipeline.py --stage all

echo ""
echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
echo ""
echo "Analytical results available in:"
echo "  s3://$S3_BUCKET/analytical-data/"
echo ""
echo "Check logs:"
echo "  ls -la ~/cts-pipeline/pipeline_logs/"
EOF

# Make executable
chmod +x ~/cts-pipeline/run-pipeline.sh

# Verify
ls -la ~/cts-pipeline/run-pipeline.sh
```

---

## Upload Raw Data

### Step 16: Prepare Local Data

On your local machine (NOT on EC2):

```bash
cd ~/path/to/CTS_Hackathon_git

# List raw data files
ls -la generated_data/

# Should show:
# attendance.csv
# events.csv
# hcp.csv
# rx_claims_monthly.csv
# data_quality_injection_log.csv
# ground_truth_config.json
```

### Step 17: Upload Raw Data to S3

On your local machine:

```bash
# Upload generated_data folder to S3
aws s3 sync generated_data/ s3://cts-hackathon-data/raw-data/ --region us-east-1

# Monitor progress
aws s3 ls s3://cts-hackathon-data/raw-data/ --recursive --human-readable

# Verify file count
aws s3 ls s3://cts-hackathon-data/raw-data/ --recursive | wc -l
```

**Should show ~6 files:**
- attendance.csv
- events.csv
- hcp.csv
- rx_claims_monthly.csv
- data_quality_injection_log.csv
- ground_truth_config.json

---

## Run the Pipeline

### Step 18: Execute Pipeline on EC2

Back on the EC2 instance:

```bash
# Go to project directory
cd ~/cts-pipeline

# Run pipeline
./run-pipeline.sh
```

**Expected output:**
```
==========================================
CTS Pipeline Execution on EC2
==========================================
S3 Mode: true
S3 Bucket: cts-hackathon-data
AWS Region: us-east-1
==========================================

SPEAKER PROGRAM ROI -- PIPELINE
  project root : /home/ec2-user/cts-pipeline
  interpreter  : /usr/bin/python3
  stage        : all
  methods      : nnm, rule_based, psm, random
  force        : False
  log file     : pipeline_logs/pipeline_run_20260818_150254.log
  timeout      : 420s per subprocess
  
  S3 mode      : ENABLED
  S3 bucket    : cts-hackathon-data
  S3 region    : us-east-1
  temp dir     : /tmp/cts-pipeline

Creating symlinks for S3 staging directories...
  Created symlink: /home/ec2-user/cts-pipeline/generated_data -> /tmp/cts-pipeline/generated_data
  Created symlink: /home/ec2-user/cts-pipeline/preprocessed_data -> /tmp/cts-pipeline/preprocessed_data
  Created symlink: /home/ec2-user/cts-pipeline/matched_pairs -> /tmp/cts-pipeline/matched_pairs
  Created symlink: /home/ec2-user/cts-pipeline/did_roi_output -> /tmp/cts-pipeline/did_roi_output

Downloading from s3://cts-hackathon-data/raw-data/ to /tmp/cts-pipeline/generated_data
  Downloaded 6 files from S3

PREPROCESSING
  [PASS] Preprocessing             45.3s  preprocess_all.py
  
MATCHING
  [PASS] Nearest Neighbor         120.1s  nnm_matching.py
  [PASS] Rule-Based                85.2s  rbm_matching.py
  [PASS] Propensity Score         180.5s  psm_matching.py
  [PASS] Random                   390.0s  random_matching.py
  
VALIDATION
  [PASS] Schema Validation          2.1s  Validated all 4 methods
  
DID/ROI ANALYSIS
  [PASS] DiD / ROI                 50.3s  did_roi_engine.py

Uploading /tmp/cts-pipeline/matched_pairs to s3://cts-hackathon-data/analytical-data/matched_pairs/
  Uploaded 12 files to S3

Uploading /tmp/cts-pipeline/did_roi_output to s3://cts-hackathon-data/analytical-data/did_roi_output/
  Uploaded 8 files to S3

Cleaning up symlinks...

PIPELINE SUMMARY
  PASS     Preprocessing            45.3s  preprocess_all.py
  PASS     Nearest Neighbor        120.1s  nnm_matching.py
  PASS     Rule-Based               85.2s  rbm_matching.py
  PASS     Propensity Score        180.5s  psm_matching.py
  PASS     Random                  390.0s  random_matching.py
  PASS     Schema Validation         2.1s  Validated all 4 methods
  PASS     DiD / ROI                50.3s  did_roi_engine.py
  total runtime: 879.3s   warnings: 0
  OVERALL: PASS

========================================
Pipeline completed successfully!
========================================

Analytical results available in:
  s3://cts-hackathon-data/analytical-data/
```

**Total runtime: ~15 minutes**

### Step 19: Monitor Pipeline Execution

If you need to check progress while pipeline runs:

**In another SSH terminal:**
```bash
# SSH into instance
ssh -i cts-deployment-key.pem ec2-user@<PUBLIC_IP>

# Watch logs
tail -f ~/cts-pipeline/pipeline_logs/pipeline_run_*.log

# Or check last lines
ls -lrt ~/cts-pipeline/pipeline_logs/
tail -100 ~/cts-pipeline/pipeline_logs/pipeline_run_*.log
```

---

## View Results

### Step 20: Verify Results in S3

On your local machine:

```bash
# List matched pairs
aws s3 ls s3://cts-hackathon-data/analytical-data/matched_pairs/ --recursive

# List DiD/ROI results
aws s3 ls s3://cts-hackathon-data/analytical-data/did_roi_output/ --recursive

# Count files
aws s3 ls s3://cts-hackathon-data/analytical-data/ --recursive | wc -l
```

**Should show ~20 result files:**
- 12 matched pairs files (4 methods × 3 files each)
- 8 DiD/ROI files (4 methods × 2 files each: results + summary)

### Step 21: Download Results Locally

```bash
# Create local results directory
mkdir -p ~/cts-results

# Download all results
aws s3 sync s3://cts-hackathon-data/analytical-data/ ~/cts-results/

# Verify download
ls -la ~/cts-results/
tree ~/cts-results/
```

### Step 22: Run Dashboard Locally (Optional)

Copy results to project directory and run dashboard:

```bash
# Copy results into project
cp -r ~/cts-results/matched_pairs ~/path/to/CTS_Hackathon_git/
cp -r ~/cts-results/did_roi_output ~/path/to/CTS_Hackathon_git/

# Run dashboard
cd ~/path/to/CTS_Hackathon_git
streamlit run dashboard.py
```

Browser will open at `http://localhost:8501`

---

## Troubleshooting

### Issue: "Connection refused" when SSHing

**Solution:**
```bash
# Check instance status in AWS Console
# Wait 2-3 minutes for instance to be fully ready
# Verify security group allows SSH (port 22)

# Try again
ssh -i cts-deployment-key.pem ec2-user@<PUBLIC_IP>
```

### Issue: "Permission denied (publickey)"

**Solution:**
```bash
# Verify key file permissions
chmod 400 cts-deployment-key.pem

# Verify key name matches
aws ec2 describe-key-pairs

# Use correct key
ssh -i cts-deployment-key.pem ec2-user@<PUBLIC_IP>
```

### Issue: "NoCredentialsError" or S3 access denied

**Solution:**
```bash
# On EC2 instance, verify IAM role
aws sts get-caller-identity

# Should show role ARN with "cts-ec2-s3-role"

# Verify S3 bucket access
aws s3 ls s3://cts-hackathon-data/

# If still failing, check bucket policy
aws s3api get-bucket-policy --bucket cts-hackathon-data
```

### Issue: Symlink creation fails on Windows

**Solution:**
This shouldn't happen on EC2 (Linux). If running locally on Windows:
1. Enable Developer Mode (Windows 10/11)
2. Or run command prompt as Administrator
3. Or use WSL2

### Issue: Pipeline runs but results not in S3

**Solution:**
```bash
# Check pipeline logs
cat ~/cts-pipeline/pipeline_logs/pipeline_run_*.log | grep -i "upload"

# Verify results exist locally
ls -la /tmp/cts-pipeline/did_roi_output/
ls -la /tmp/cts-pipeline/matched_pairs/

# Manually upload if needed
aws s3 sync /tmp/cts-pipeline/matched_pairs s3://cts-hackathon-data/analytical-data/matched_pairs/
aws s3 sync /tmp/cts-pipeline/did_roi_output s3://cts-hackathon-data/analytical-data/did_roi_output/
```

### Issue: Out of disk space

**Solution:**
```bash
# Check disk usage
df -h

# Clean up temporary files
rm -rf /tmp/cts-pipeline/

# If needed, expand volume (AWS Console)
```

### Issue: Pipeline runs slowly

**Solution:**
```bash
# Check instance type (should be t3.large or larger)
aws ec2 describe-instances --query 'Reservations[].Instances[].InstanceType'

# Monitor CPU/Memory during run
top
# Press 'q' to quit

# If too slow, stop instance and change type to t3.xlarge
```

---

## Cost Estimation

**Monthly costs (approximate):**

| Service | Usage | Cost |
|---------|-------|------|
| EC2 (t3.large) | 24/7 usage | ~$50-60 |
| S3 Storage | 100 MB data | ~$0.25 |
| S3 Requests | 1000s reads/writes | ~$0.50 |
| Data Transfer | Minimal (within AWS) | ~$0 |
| **Total** | | **~$50-60/month** |

**To reduce costs:**
- Use t3.micro for testing (free tier eligible)
- Stop instance when not in use
- Archive results to Glacier after 30 days
- Use S3 Lifecycle policies

---

## Cleanup (Stop/Terminate)

### Stop Instance (Pause billing)

**Via AWS Console:**
1. Go to EC2 → Instances
2. Right-click instance → Instance State → Stop
3. Instance stops (data preserved, minimal cost)

**Via AWS CLI:**
```bash
aws ec2 stop-instances --instance-ids i-1234567890abcdef0
```

### Terminate Instance (Delete)

**Via AWS CLI:**
```bash
aws ec2 terminate-instances --instance-ids i-1234567890abcdef0
```

### Delete S3 Bucket (Optional)

```bash
# Delete all objects
aws s3 rm s3://cts-hackathon-data --recursive

# Delete bucket
aws s3 rb s3://cts-hackathon-data
```

---

## Summary

✅ **Step-by-step checklist:**

- [ ] Create S3 bucket with raw-data/ and analytical-data/ folders
- [ ] Create IAM role and instance profile
- [ ] Create EC2 key pair
- [ ] Launch t3.large EC2 instance with IAM role
- [ ] SSH into instance
- [ ] Install Python 3, pip, git
- [ ] Clone project repository
- [ ] Install Python dependencies
- [ ] Configure .env file with S3 credentials
- [ ] Upload raw data to S3
- [ ] Run pipeline: `./run-pipeline.sh`
- [ ] Verify results in S3
- [ ] Download results locally
- [ ] (Optional) Run dashboard on local machine

**Total time: ~2-3 hours (including pipeline execution)**

---

## Next Steps

1. **Deploy**: Follow steps 1-19 above
2. **Run**: Execute `./run-pipeline.sh` on EC2
3. **Analyze**: Download results and use dashboard
4. **Iterate**: Modify parameters and re-run
5. **Scale**: Use larger instances for production

---

## Support

For issues or questions:
1. Check **S3_SETUP.md** for additional S3 details
2. Check **pipeline_logs/** for detailed error messages
3. Review AWS Console for resource status
4. Check EC2 security group rules

**Happy deploying! 🚀**
