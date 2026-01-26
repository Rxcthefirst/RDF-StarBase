# =============================================================================
# RDF-StarBase Quick Deploy Script (PowerShell)
# =============================================================================
# Usage: .\deploy.ps1 [-Region us-east-1]
# =============================================================================

param(
    [string]$Region = "us-east-1"
)

$ErrorActionPreference = "Stop"

$ClusterName = "rdf-starbase-cluster"
$ServiceName = "rdf-starbase-service"
$TaskFamily = "rdf-starbase"
$Image = "rxcthefirst/rdf-starbase:latest"
$LogGroup = "/ecs/rdf-starbase"

Write-Host "=== RDF-StarBase ECS Deployment ===" -ForegroundColor Cyan
Write-Host "Region: $Region"
Write-Host ""

# Check AWS CLI
try {
    $identity = aws sts get-caller-identity --output json | ConvertFrom-Json
    $AccountId = $identity.Account
    Write-Host "AWS Account: $AccountId"
} catch {
    Write-Host "ERROR: AWS CLI not configured. Run 'aws configure' first." -ForegroundColor Red
    exit 1
}

# Create log group
Write-Host "Creating CloudWatch log group..." -ForegroundColor Yellow
aws logs create-log-group --log-group-name $LogGroup --region $Region 2>$null
if (-not $?) { Write-Host "  (already exists)" }

# Create ECS cluster
Write-Host "Creating ECS cluster..." -ForegroundColor Yellow
aws ecs create-cluster --cluster-name $ClusterName --region $Region --settings name=containerInsights,value=enabled 2>$null
if (-not $?) { Write-Host "  (already exists)" }

# Check execution role
Write-Host "Checking IAM execution role..." -ForegroundColor Yellow
$roleExists = aws iam get-role --role-name ecsTaskExecutionRole 2>$null
if (-not $roleExists) {
    Write-Host "Creating IAM execution role..." -ForegroundColor Yellow
    
    $trustPolicy = @{
        Version = "2012-10-17"
        Statement = @(@{
            Effect = "Allow"
            Principal = @{ Service = "ecs-tasks.amazonaws.com" }
            Action = "sts:AssumeRole"
        })
    } | ConvertTo-Json -Depth 10 -Compress
    
    aws iam create-role `
        --role-name ecsTaskExecutionRole `
        --assume-role-policy-document $trustPolicy
    
    aws iam attach-role-policy `
        --role-name ecsTaskExecutionRole `
        --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
}

# Get VPC info
Write-Host "Getting VPC configuration..." -ForegroundColor Yellow
$VpcId = aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text --region $Region
$Subnets = (aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VpcId" --query "Subnets[*].SubnetId" --output text --region $Region) -split "`t"

# Create security group
Write-Host "Creating security group..." -ForegroundColor Yellow
$SgId = aws ec2 describe-security-groups --filters "Name=group-name,Values=rdf-starbase-sg" --query "SecurityGroups[0].GroupId" --output text --region $Region 2>$null

if ($SgId -eq "None" -or -not $SgId) {
    $SgId = aws ec2 create-security-group `
        --group-name rdf-starbase-sg `
        --description "RDF-StarBase ECS security group" `
        --vpc-id $VpcId `
        --region $Region `
        --query 'GroupId' `
        --output text
}

# Add ingress rule
aws ec2 authorize-security-group-ingress `
    --group-id $SgId `
    --protocol tcp `
    --port 8000 `
    --cidr 0.0.0.0/0 `
    --region $Region 2>$null

# Register task definition
Write-Host "Registering task definition..." -ForegroundColor Yellow
$ExecutionRoleArn = "arn:aws:iam::${AccountId}:role/ecsTaskExecutionRole"

$containerDefs = @(@{
    name = "rdf-starbase"
    image = $Image
    essential = $true
    portMappings = @(@{
        containerPort = 8000
        protocol = "tcp"
    })
    environment = @(
        @{ name = "RDFSTARBASE_SERVE_STATIC"; value = "true" }
        @{ name = "PYTHONUNBUFFERED"; value = "1" }
    )
    logConfiguration = @{
        logDriver = "awslogs"
        options = @{
            "awslogs-group" = $LogGroup
            "awslogs-region" = $Region
            "awslogs-stream-prefix" = "ecs"
        }
    }
    healthCheck = @{
        command = @("CMD-SHELL", "curl -f http://localhost:8000/health || exit 1")
        interval = 30
        timeout = 5
        retries = 3
        startPeriod = 60
    }
}) | ConvertTo-Json -Depth 10 -Compress

aws ecs register-task-definition `
    --family $TaskFamily `
    --network-mode awsvpc `
    --requires-compatibilities FARGATE `
    --cpu 512 `
    --memory 1024 `
    --execution-role-arn $ExecutionRoleArn `
    --container-definitions $containerDefs `
    --region $Region

# Create or update service
Write-Host "Creating/updating ECS service..." -ForegroundColor Yellow
$SubnetList = ($Subnets[0..1]) -join ","
$NetworkConfig = "awsvpcConfiguration={subnets=[$SubnetList],securityGroups=[$SgId],assignPublicIp=ENABLED}"

$serviceExists = aws ecs describe-services --cluster $ClusterName --services $ServiceName --region $Region --query "services[0].status" --output text 2>$null

if ($serviceExists -eq "ACTIVE") {
    aws ecs update-service `
        --cluster $ClusterName `
        --service $ServiceName `
        --task-definition $TaskFamily `
        --force-new-deployment `
        --region $Region
} else {
    aws ecs create-service `
        --cluster $ClusterName `
        --service-name $ServiceName `
        --task-definition $TaskFamily `
        --desired-count 1 `
        --launch-type FARGATE `
        --network-configuration $NetworkConfig `
        --region $Region
}

Write-Host ""
Write-Host "=== Deployment initiated! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Wait for the task to start (~1-2 min), then get the public IP:" -ForegroundColor Cyan
Write-Host "  `$taskArn = aws ecs list-tasks --cluster $ClusterName --region $Region --query 'taskArns[0]' --output text"
Write-Host "  aws ecs describe-tasks --cluster $ClusterName --tasks `$taskArn --region $Region --query 'tasks[0].attachments[0].details'"
Write-Host ""
Write-Host "Or use this one-liner to get the public IP:"
Write-Host "  `$eni = aws ecs describe-tasks --cluster $ClusterName --tasks (aws ecs list-tasks --cluster $ClusterName --query 'taskArns[0]' --output text --region $Region) --region $Region --query 'tasks[0].attachments[0].details[?name==``networkInterfaceId``].value' --output text"
Write-Host "  aws ec2 describe-network-interfaces --network-interface-ids `$eni --query 'NetworkInterfaces[0].Association.PublicIp' --output text --region $Region"
Write-Host ""
Write-Host "Access the app at: http://<PUBLIC_IP>:8000/app/"
Write-Host "API docs at: http://<PUBLIC_IP>:8000/docs"
Write-Host ""
Write-Host "View logs:"
Write-Host "  aws logs tail $LogGroup --follow --region $Region"
