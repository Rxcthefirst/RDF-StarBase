#!/bin/bash
# =============================================================================
# RDF-StarBase Quick Deploy Script
# =============================================================================
# Usage: ./deploy.sh [region]
# Example: ./deploy.sh us-east-1
# =============================================================================

set -e

REGION=${1:-us-east-1}
CLUSTER_NAME="rdf-starbase-cluster"
SERVICE_NAME="rdf-starbase-service"
TASK_FAMILY="rdf-starbase"
IMAGE="rxcthefirst/rdf-starbase:latest"
LOG_GROUP="/ecs/rdf-starbase"

echo "=== RDF-StarBase ECS Deployment ==="
echo "Region: $REGION"
echo ""

# Check if AWS CLI is configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo "ERROR: AWS CLI not configured. Run 'aws configure' first."
    exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "AWS Account: $ACCOUNT_ID"

# Create log group
echo "Creating CloudWatch log group..."
aws logs create-log-group --log-group-name $LOG_GROUP --region $REGION 2>/dev/null || true

# Create ECS cluster
echo "Creating ECS cluster..."
aws ecs create-cluster --cluster-name $CLUSTER_NAME --region $REGION --settings name=containerInsights,value=enabled || true

# Create execution role if it doesn't exist
echo "Checking IAM execution role..."
if ! aws iam get-role --role-name ecsTaskExecutionRole &> /dev/null; then
    echo "Creating IAM execution role..."
    aws iam create-role \
        --role-name ecsTaskExecutionRole \
        --assume-role-policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }'
    
    aws iam attach-role-policy \
        --role-name ecsTaskExecutionRole \
        --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
fi

# Get default VPC info
echo "Getting VPC configuration..."
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text --region $REGION)
SUBNETS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query "Subnets[*].SubnetId" --output text --region $REGION | tr '\t' ',')

# Create security group
echo "Creating security group..."
SG_ID=$(aws ec2 create-security-group \
    --group-name rdf-starbase-sg \
    --description "RDF-StarBase ECS security group" \
    --vpc-id $VPC_ID \
    --region $REGION \
    --query 'GroupId' \
    --output text 2>/dev/null) || SG_ID=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=rdf-starbase-sg" --query "SecurityGroups[0].GroupId" --output text --region $REGION)

# Add ingress rule
aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp \
    --port 8000 \
    --cidr 0.0.0.0/0 \
    --region $REGION 2>/dev/null || true

# Register task definition
echo "Registering task definition..."
EXECUTION_ROLE_ARN="arn:aws:iam::$ACCOUNT_ID:role/ecsTaskExecutionRole"

aws ecs register-task-definition \
    --family $TASK_FAMILY \
    --network-mode awsvpc \
    --requires-compatibilities FARGATE \
    --cpu 512 \
    --memory 1024 \
    --execution-role-arn $EXECUTION_ROLE_ARN \
    --container-definitions "[
        {
            \"name\": \"rdf-starbase\",
            \"image\": \"$IMAGE\",
            \"essential\": true,
            \"portMappings\": [{\"containerPort\": 8000, \"protocol\": \"tcp\"}],
            \"environment\": [
                {\"name\": \"RDFSTARBASE_SERVE_STATIC\", \"value\": \"true\"},
                {\"name\": \"PYTHONUNBUFFERED\", \"value\": \"1\"}
            ],
            \"logConfiguration\": {
                \"logDriver\": \"awslogs\",
                \"options\": {
                    \"awslogs-group\": \"$LOG_GROUP\",
                    \"awslogs-region\": \"$REGION\",
                    \"awslogs-stream-prefix\": \"ecs\"
                }
            },
            \"healthCheck\": {
                \"command\": [\"CMD-SHELL\", \"curl -f http://localhost:8000/health || exit 1\"],
                \"interval\": 30,
                \"timeout\": 5,
                \"retries\": 3,
                \"startPeriod\": 60
            }
        }
    ]" \
    --region $REGION

# Create or update service
echo "Creating/updating ECS service..."
FIRST_TWO_SUBNETS=$(echo $SUBNETS | cut -d',' -f1-2)

aws ecs create-service \
    --cluster $CLUSTER_NAME \
    --service-name $SERVICE_NAME \
    --task-definition $TASK_FAMILY \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[$FIRST_TWO_SUBNETS],securityGroups=[$SG_ID],assignPublicIp=ENABLED}" \
    --region $REGION 2>/dev/null || \
aws ecs update-service \
    --cluster $CLUSTER_NAME \
    --service $SERVICE_NAME \
    --task-definition $TASK_FAMILY \
    --force-new-deployment \
    --region $REGION

echo ""
echo "=== Deployment initiated! ==="
echo ""
echo "Wait for the task to start, then get the public IP:"
echo "  aws ecs list-tasks --cluster $CLUSTER_NAME --region $REGION"
echo "  aws ecs describe-tasks --cluster $CLUSTER_NAME --tasks <task-arn> --region $REGION"
echo ""
echo "Access the app at: http://<PUBLIC_IP>:8000/app/"
echo "API docs at: http://<PUBLIC_IP>:8000/docs"
echo ""
echo "View logs:"
echo "  aws logs tail $LOG_GROUP --follow --region $REGION"
