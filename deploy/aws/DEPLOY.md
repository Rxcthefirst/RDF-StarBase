# =============================================================================
# RDF-StarBase AWS ECS Deployment Guide
# =============================================================================

This guide covers deploying RDF-StarBase to AWS ECS Fargate.

## Prerequisites

- AWS CLI installed and configured (`aws configure`)
- Terraform >= 1.0 installed
- Docker image published to Docker Hub (already done: `rxcthefirst/rdf-starbase:latest`)

## Quick Deploy with Terraform

### 1. Navigate to the Terraform directory

```bash
cd deploy/aws/terraform
```

### 2. Initialize Terraform

```bash
terraform init
```

### 3. Review the plan

```bash
terraform plan
```

### 4. Deploy

```bash
terraform apply
```

Type `yes` when prompted. Deployment takes approximately 3-5 minutes.

### 5. Access your application

After deployment completes, Terraform outputs the URLs:

```
app_url = "http://rdf-starbase-alb-xxxxxx.us-east-1.elb.amazonaws.com/app/"
api_docs_url = "http://rdf-starbase-alb-xxxxxx.us-east-1.elb.amazonaws.com/docs"
```

## Manual Deploy with AWS CLI

If you prefer not to use Terraform, here are the manual steps:

### 1. Create CloudWatch Log Group

```bash
aws logs create-log-group --log-group-name /ecs/rdf-starbase --region us-east-1
```

### 2. Create ECS Cluster

```bash
aws ecs create-cluster --cluster-name rdf-starbase-cluster --region us-east-1
```

### 3. Create IAM Execution Role

```bash
# Create the role
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

# Attach the policy
aws iam attach-role-policy \
  --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
```

### 4. Register Task Definition

Edit `ecs-task-definition.json` to replace placeholders, then:

```bash
aws ecs register-task-definition \
  --cli-input-json file://ecs-task-definition.json \
  --region us-east-1
```

### 5. Create Service with ALB

```bash
# Get your VPC, subnet, and security group IDs first
aws ec2 describe-vpcs --query "Vpcs[0].VpcId" --output text
aws ec2 describe-subnets --query "Subnets[*].SubnetId" --output text

# Create the service
aws ecs create-service \
  --cluster rdf-starbase-cluster \
  --service-name rdf-starbase-service \
  --task-definition rdf-starbase \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx,subnet-yyy],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" \
  --region us-east-1
```

## Updating the Deployment

### Update Docker Image

```bash
# Build and push new image
docker build -t rxcthefirst/rdf-starbase:latest .
docker push rxcthefirst/rdf-starbase:latest

# Force new deployment
aws ecs update-service \
  --cluster rdf-starbase-cluster \
  --service rdf-starbase-service \
  --force-new-deployment \
  --region us-east-1
```

### Scale Up/Down

```bash
aws ecs update-service \
  --cluster rdf-starbase-cluster \
  --service rdf-starbase-service \
  --desired-count 2 \
  --region us-east-1
```

## Monitoring

### View Logs

```bash
aws logs tail /ecs/rdf-starbase --follow --region us-east-1
```

### Check Service Status

```bash
aws ecs describe-services \
  --cluster rdf-starbase-cluster \
  --services rdf-starbase-service \
  --region us-east-1 \
  --query "services[0].{status:status,running:runningCount,desired:desiredCount}"
```

### View Running Tasks

```bash
aws ecs list-tasks \
  --cluster rdf-starbase-cluster \
  --service-name rdf-starbase-service \
  --region us-east-1
```

## Cleanup

### With Terraform

```bash
cd deploy/aws/terraform
terraform destroy
```

### With AWS CLI

```bash
# Delete service
aws ecs update-service --cluster rdf-starbase-cluster --service rdf-starbase-service --desired-count 0 --region us-east-1
aws ecs delete-service --cluster rdf-starbase-cluster --service rdf-starbase-service --region us-east-1

# Delete cluster
aws ecs delete-cluster --cluster rdf-starbase-cluster --region us-east-1

# Delete log group
aws logs delete-log-group --log-group-name /ecs/rdf-starbase --region us-east-1
```

## Cost Estimate

For a demo environment (Fargate, 0.5 vCPU, 1GB RAM, 1 task):
- **ECS Fargate**: ~$15-20/month
- **ALB**: ~$20/month
- **EFS**: ~$0.30/GB/month (minimal for demo)
- **Data Transfer**: Varies by usage

**Total**: ~$35-50/month for a basic demo deployment

## Troubleshooting

### Task fails to start

1. Check CloudWatch logs:
   ```bash
   aws logs tail /ecs/rdf-starbase --region us-east-1
   ```

2. Verify security group allows outbound traffic for Docker pulls

3. Ensure EFS mount targets are in the same subnets as tasks

### Health check failures

1. Verify the `/health` endpoint responds:
   ```bash
   curl http://ALB_DNS_NAME/health
   ```

2. Check security group allows traffic from ALB to ECS tasks on port 8000

### Container exits immediately

1. Check if the image exists:
   ```bash
   docker pull rxcthefirst/rdf-starbase:latest
   ```

2. Test locally first:
   ```bash
   docker run -p 8000:8000 rxcthefirst/rdf-starbase:latest
   ```
