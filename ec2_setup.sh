#!/bin/bash
# EC2 Deployment Setup Script for CTS Hackathon Pipeline with S3
# Usage: bash ec2_setup.sh

set -e  # Exit on error

echo "================================================"
echo "CTS Hackathon Pipeline - EC2 Setup"
echo "================================================"

# Configuration (Modify these)
S3_BUCKET="${1:-cts-hackathon-data}"
AWS_REGION="${2:-us-east-1}"
PROJECT_DIR="/home/ubuntu/cts-pipeline"

echo "✓ S3 Bucket: $S3_BUCKET"
echo "✓ AWS Region: $AWS_REGION"
echo "✓ Project Directory: $PROJECT_DIR"
echo ""

# Step 1: Update system
echo "Step 1: Updating system packages..."
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip git

# Step 2: Install Python dependencies
echo "Step 2: Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install boto3 numpy pandas scikit-learn statsmodels streamlit plotly Faker

# Step 3: Clone repository (or use existing)
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Step 3: Cloning repository..."
    git clone https://github.com/YOUR_REPO/cts-pipeline.git "$PROJECT_DIR"
else
    echo "Step 3: Repository already exists at $PROJECT_DIR"
    cd "$PROJECT_DIR"
    git pull origin main
fi

cd "$PROJECT_DIR"

# Step 4: Create .env file for S3 configuration
echo "Step 4: Creating .env file for S3 configuration..."
cat > .env << EOF
# AWS S3 Configuration
S3_ENABLED=true
S3_BUCKET=$S3_BUCKET
S3_REGION=$AWS_REGION

# Temporary directory for pipeline staging
TEMP_DIR=/tmp/cts-pipeline
EOF

echo "✓ .env file created"

# Step 5: Create startup script
echo "Step 5: Creating pipeline execution script..."
cat > run_pipeline.sh << 'SCRIPT'
#!/bin/bash
set -e

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

echo "=========================================="
echo "CTS Pipeline Execution"
echo "=========================================="
echo "S3 Mode: $S3_ENABLED"
echo "S3 Bucket: $S3_BUCKET"
echo "AWS Region: $S3_REGION"
echo "=========================================="
echo ""

# Run pipeline
python3 pipeline.py --stage all

echo ""
echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
echo ""
echo "Analytical results available in:"
echo "  s3://$S3_BUCKET/analytical-data/"
echo ""
echo "Pipeline logs:"
echo "  $(ls -1t pipeline_logs/pipeline_report_*.md | head -1)"
SCRIPT

chmod +x run_pipeline.sh
echo "✓ Pipeline execution script created: run_pipeline.sh"

# Step 6: Verify AWS credentials
echo "Step 6: Verifying AWS credentials..."
if aws sts get-caller-identity > /dev/null 2>&1; then
    echo "✓ AWS credentials configured successfully"
    aws sts get-caller-identity
else
    echo "✗ AWS credentials not configured"
    echo "  Configure with: aws configure"
fi

# Step 7: Verify S3 bucket access
echo ""
echo "Step 7: Verifying S3 bucket access..."
if aws s3 ls "s3://$S3_BUCKET" --region "$AWS_REGION" > /dev/null 2>&1; then
    echo "✓ S3 bucket accessible: s3://$S3_BUCKET"
    echo "  Bucket contents:"
    aws s3 ls "s3://$S3_BUCKET" --recursive --human-readable --summarize --region "$AWS_REGION" | head -20
else
    echo "✗ S3 bucket not accessible"
    echo "  Please verify:"
    echo "    - Bucket name: $S3_BUCKET"
    echo "    - AWS region: $AWS_REGION"
    echo "    - IAM permissions"
    echo "    - AWS credentials are configured"
fi

# Step 8: Display next steps
echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Next Steps:"
echo "1. Upload raw data to S3:"
echo "   aws s3 sync ./generated_data s3://$S3_BUCKET/raw-data/"
echo ""
echo "2. Run the pipeline:"
echo "   ./run_pipeline.sh"
echo ""
echo "3. Monitor execution:"
echo "   tail -f pipeline_logs/pipeline_run_*.log"
echo ""
echo "4. Check results in S3:"
echo "   aws s3 ls s3://$S3_BUCKET/analytical-data/ --recursive"
echo ""
echo "5. Run dashboard:"
echo "   streamlit run dashboard.py"
echo ""
