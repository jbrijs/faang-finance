FROM public.ecr.aws/lambda/python:3.12

# Copy requirements.txt into the LAMBDA_TASK_ROOT directory
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install Python packages in the LAMBDA_TASK_ROOT directory
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy all files from ./src into the LAMBDA_TASK_ROOT directory
COPY src/ ${LAMBDA_TASK_ROOT}/

# Set the CMD to the handler function defined in main.py
CMD [ "main.lambda_handler" ]
