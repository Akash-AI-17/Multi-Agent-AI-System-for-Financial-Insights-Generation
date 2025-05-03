from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
from dotenv import load_dotenv
import time
import threading
import math
import asyncio
from typing import Dict, Any
import uvicorn

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai_tools import PDFSearchTool
from langchain_openai import ChatOpenAI
from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_exception_type
import litellm
from langchain.cache import InMemoryCache
import langchain

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
print(f"Loaded GROQ_API_KEY: {'Yes' if api_key else 'No'}")

# Set up LangChain caching
langchain.llm_cache = InMemoryCache()

# Initialize FastAPI app
app = FastAPI(title="Transcript Analysis API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dictionary to store task statuses
task_statuses: Dict[str, Dict[str, Any]] = {}

# Create directories
os.makedirs("uploads", exist_ok=True)

# Create a token bucket rate limiter with a global lock
class TokenBucketRateLimiter:
    def __init__(self, tokens_per_minute=4500):  # Set much lower than the 6000 limit
        self.tokens_per_minute = tokens_per_minute
        self.tokens_per_second = tokens_per_minute / 60
        self.tokens = tokens_per_minute  # Start with a full bucket
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        refill_amount = elapsed * self.tokens_per_second

        with self.lock:
            self.tokens = min(self.tokens_per_minute, self.tokens + refill_amount)
            self.last_refill = now

    def consume(self, tokens):
        self.refill()  # Refill based on elapsed time

        with self.lock:
            if tokens > self.tokens:
                # Calculate wait time needed to accumulate enough tokens
                wait_time = (tokens - self.tokens) / self.tokens_per_second
                wait_time = math.ceil(wait_time) + 2  # Add buffer
                print(f"⏳ Rate limit prevention: Waiting {wait_time}s to accumulate tokens")
                time.sleep(wait_time)
                self.refill()  # Refill after waiting

            # Double-check we have enough tokens now
            if tokens > self.tokens:
                print(f"⚠️ Still not enough tokens. Have {self.tokens}, need {tokens}")
                return False

            self.tokens -= tokens
            print(f"🪙 Consumed {tokens} tokens. Remaining: {self.tokens:.2f}/{self.tokens_per_minute}")
            return True


# Create a global rate limiter
rate_limiter = TokenBucketRateLimiter(tokens_per_minute=4500)


class RateLimitedGroqLLM:
    def __init__(self, api_key, model_name="deepseek-r1-distill-llama-70b", max_tokens=200, temperature=0.4):
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.last_request_time = 0

    @retry(
        retry=retry_if_exception_type((litellm.RateLimitError, Exception)),
        wait=wait_fixed(10),  # Fixed 10-second wait on errors
        stop=stop_after_attempt(5)
    )
    def _call_api_with_retry(self, llm_instance, **kwargs):
        # Enforce minimum time between requests (15 seconds)
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < 15:
            sleep_time = 15 - time_since_last
            print(f"😴 Enforcing minimum delay: {sleep_time:.2f}s between requests")
            time.sleep(sleep_time)

        # Estimate tokens for request (rough estimate)
        # Calculate based on prompt length and expected output tokens
        prompt_tokens = len(str(kwargs.get("messages", ""))) // 3
        output_tokens = kwargs.get("max_tokens", self.max_tokens)
        estimated_tokens = prompt_tokens + output_tokens

        # Consume tokens from rate limiter
        rate_limiter.consume(estimated_tokens)

        print(f"🚀 Making API request using model {self.model_name}")
        result = llm_instance(**kwargs)

        # Update last request time
        self.last_request_time = time.time()

        # Add mandatory delay after every request
        time.sleep(5)

        return result

    def get_langchain_llm(self):
        return ChatOpenAI(
            openai_api_base="https://api.groq.com/openai/v1",
            openai_api_key=self.api_key,
            model_name=f"groq/{self.model_name}",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            request_timeout=180  # Longer timeout to account for rate limit waits
        )

    def __call__(self, **kwargs):
        llm_instance = self.get_langchain_llm()
        return self._call_api_with_retry(llm_instance, **kwargs)


def slow_step_callback(step):
    print(f"Completed step: {step}")
    time.sleep(10)  # Add 10-second delay between steps
    return step


async def process_transcript(task_id: str, file_path: str):
    """Process transcript using CrewAI in the background"""
    try:
        # Update task status
        task_statuses[task_id] = {"status": "processing", "result": None}

        # Create LLM instances
        primary_llm = RateLimitedGroqLLM(
            api_key=api_key,
            model_name="deepseek-r1-distill-llama-70b",
            max_tokens=200,
            temperature=0.4
        ).get_langchain_llm()

        smaller_llm = RateLimitedGroqLLM(
            api_key=api_key,
            model_name="llama-3.1-8b-instant",
            max_tokens=400,
            temperature=0.4
        ).get_langchain_llm()

        # Create RAG tool
        rag_tool = PDFSearchTool(
            pdf=file_path,
            config=dict(
                llm=dict(
                    provider="groq",
                    config=dict(
                        model="llama-3.1-8b-instant",
                        max_tokens=150,
                    ),
                ),
                embedder=dict(
                    provider="huggingface",
                    config=dict(
                        model="sentence-transformers/all-MiniLM-L6-v2",
                    ),
                ),
            )
        )

        # Create agents
        transcript_analyst = Agent(
            role="Senior Equity Research Analyst",
            goal="Extract comprehensive investment insights from transcripts to support accurate reporting.",
            backstory="""
            You're a former hedge fund analyst with 12 years of experience parsing executive commentary.
            You excel at identifying forward-looking statements, performance data, and management tone.
            Be thorough but focus on key insights only.
            """,
            tools=[rag_tool],
            verbose=True,
            llm=primary_llm,
            max_iter=2,
            max_rpm=2,
            step_callback=slow_step_callback
        )

        investment_strategist = Agent(
            role="Chief Investment Strategist",
            goal="Create a clear, defensible investment thesis based on transcript analysis.",
            backstory="""
            You're a veteran buy-side strategist trusted by portfolio managers for your reliable insights.
            You distill key drivers, risks, and valuation context from management commentary.
            Base all insights strictly on extracted information.
            """,
            tools=[rag_tool],
            verbose=True,
            llm=primary_llm,
            max_iter=2,
            max_rpm=2,
            step_callback=slow_step_callback
        )

        financial_writer = Agent(
            role="Institutional-Grade Financial Writer",
            goal="Write a professional, comprehensive article for institutional investors based on the provided analysis.",
            backstory="""
            You're a former Bloomberg and Reuters journalist writing in the style of WSJ and Financial Times.
            Your audience includes institutional investors who expect factual, well-structured content.
            Your articles are known for clarity, depth, and professional tone.
            """,
            verbose=True,
            llm=smaller_llm,
            tools=[rag_tool],
            max_iter=2,
            max_rpm=2,
            step_callback=slow_step_callback
        )

        # Create tasks
        extraction_task = Task(
            description="""
            ANALYZE THE TRANSCRIPT: Extract key investment information including:
            1. Forward-looking statements with timelines
            2. Management's CAPEX and expansion plans
            3. Guidance revisions
            4. Management tone and confidence
            5. Segment-wise performance mentions
            6. Extract Company Name mentioned in transcript header or intro (e.g., "TATA Motors")
            7. Discussion of risks, debt, and liquidity

            Be comprehensive but focus on the most important insights and only extract what's explicitly stated.
            Use direct quotes where possible and note who said what.
            If something is unclear or missing, mark it as "Not Discussed."
            """,
            agent=transcript_analyst,
            expected_output="""
            A structured list of key investment insights with:
            - Direct quotes and speaker attribution when available
            - Confidence levels for each insight
            - Company Name
            """
        )

        analysis_task = Task(
            description="""
            Use the extracted insights to build a structured investment thesis.

            Include:
            1. Investment Summary: High-level view of opportunity or risk
            2. Key Upside Catalysts: Ranked by importance
            3. Key Risk Factors: Near-term and structural
            4. Outlook: Based on management's comments

            If specific information is missing, indicate this rather than making assumptions.
            Be thorough but concise.
            """,
            agent=investment_strategist,
            expected_output="""
            A comprehensive investment thesis that clearly separates known facts from areas lacking information.
            Focus on providing actionable insights for the final article.
            """,
            context=[extraction_task]
        )

        writing_task = Task(
            description="""
            Write a comprehensive article titled "[Company Name]: Latest Developments and Outlook" with the following sections:

            1. Executive Summary (brief overview of key points)
            2. Management Tone, Commentary & Strategic Direction
            3. Financial Performance Highlights (give financial numbers wisely. If number seems ambiguous then avoid it.)
            4. Segment-wise Performance (if available)
            5. Forward Guidance & Growth Outlook
            6. Investment Risks & Considerations
            7. Conclusion

            Only include sections where you have sufficient information from the analysis.
            For any major topic not covered in the transcript, simply omit that section rather than speculating.

            The article should be professional, fact-based, and properly structured with headers for each section.
            Aim for approximately 1,000-1,200 words total.

            Use markdown formatting for better readability.
            """,
            agent=financial_writer,
            context=[analysis_task],
            expected_output="""
            A professional, well-structured article in markdown format covering all relevant aspects 
            of the company based on the transcript analysis. Each section should be properly formatted 
            with headers and paragraphs.
            """
        )

        # Create crew
        crew = Crew(
            agents=[transcript_analyst, investment_strategist, financial_writer],
            tasks=[extraction_task, analysis_task, writing_task],
            verbose=True,
            process=Process.sequential,
        )

        # Pre-execution delay
        time.sleep(5)

        # Run the analysis
        print(f"Starting CrewAI processing for task {task_id}")
        result = crew.kickoff()
        print(f"CrewAI processing complete for task {task_id}")

        # Extract the result text from CrewOutput object
        if hasattr(result, 'raw'):
            result_text = result.raw
        else:
            result_text = str(result)  # Fallback to string conversion

        # Update task status with result text
        task_statuses[task_id] = {"status": "completed", "result": result_text}

    except Exception as e:
        print(f"Error processing task {task_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        task_statuses[task_id] = {"status": "failed", "error": str(e)}
    finally:
        # Keep the file for debugging purposes but you might want to remove it in production
        print(f"Analysis complete for file: {file_path}")


# Homepage with file upload form
@app.get("/", response_class=HTMLResponse)
async def get_form():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Financial Transcript Analysis</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
            }
            h1, h2 {
                color: #2c3e50;
            }
            .container {
                background-color: #f9f9f9;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: bold;
                color: #34495e;
            }
            input[type="file"] {
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                width: 100%;
                background-color: white;
            }
            button {
                background-color: #3498db;
                color: white;
                padding: 12px 20px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                transition: background-color 0.3s;
            }
            button:hover {
                background-color: #2980b9;
            }
            #results {
                margin-top: 30px;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 4px;
                display: none;
                background-color: white;
            }
            #status {
                font-weight: bold;
                color: #e74c3c;
            }
            .note {
                font-size: 14px;
                color: #7f8c8d;
                margin-top: 10px;
            }
            pre {
                white-space: pre-wrap;
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 4px;
                padding: 15px;
                overflow-x: auto;
            }
            .loader {
                border: 5px solid #f3f3f3;
                border-top: 5px solid #3498db;
                border-radius: 50%;
                width: 30px;
                height: 30px;
                animation: spin 2s linear infinite;
                display: inline-block;
                margin-left: 10px;
                vertical-align: middle;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Financial Transcript Analysis</h1>
            <p>Upload an earnings call transcript PDF to generate a professional investment analysis article.</p>

            <div class="form-group">
                <label for="pdfFile">PDF Transcript:</label>
                <input type="file" id="pdfFile" accept="application/pdf" />
                <p class="note">Supported format: PDF files of earnings call transcripts</p>
            </div>

            <button id="analyzeBtn" onclick="uploadFile()">Analyze Transcript</button>

            <div id="results">
                <h2>Analysis Results</h2>
                <p>Task ID: <span id="taskId"></span></p>
                <p>Status: <span id="status">Uploading...</span> <span id="loader" class="loader"></span></p>
                <div id="resultContent" style="margin-top: 20px;"></div>
            </div>
        </div>

        <script>
            function uploadFile() {
                const fileInput = document.getElementById('pdfFile');
                const file = fileInput.files[0];
                const analyzeBtn = document.getElementById('analyzeBtn');

                if (!file) {
                    alert('Please select a PDF file');
                    return;
                }

                // Disable button during processing
                analyzeBtn.disabled = true;
                analyzeBtn.textContent = 'Uploading...';

                // Display results div
                document.getElementById('results').style.display = 'block';
                document.getElementById('status').textContent = 'Uploading...';
                document.getElementById('loader').style.display = 'inline-block';

                // Create form data
                const formData = new FormData();
                formData.append('transcript_file', file);

                // Upload file
                fetch('/analyze/', {
                    method: 'POST',
                    body: formData
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => {
                    document.getElementById('taskId').textContent = data.task_id;
                    document.getElementById('status').textContent = 'Processing...';

                    // Start polling for status
                    pollStatus(data.task_id);
                })
                .catch(error => {
                    document.getElementById('status').textContent = 'Error: ' + error.message;
                    document.getElementById('loader').style.display = 'none';
                    analyzeBtn.disabled = false;
                    analyzeBtn.textContent = 'Analyze Transcript';
                });
            }

            function pollStatus(taskId) {
                const interval = setInterval(() => {
                    fetch(`/task-status/${taskId}`)
                    .then(response => {
                        if (!response.ok) {
                            throw new Error(`HTTP error! Status: ${response.status}`);
                        }
                        return response.json();
                    })
                    .then(data => {
                        document.getElementById('status').textContent = data.status;

                        if (data.status === 'completed') {
                            clearInterval(interval);
                            document.getElementById('loader').style.display = 'none';
                            document.getElementById('analyzeBtn').disabled = false;
                            document.getElementById('analyzeBtn').textContent = 'Analyze Transcript';

                            // Format the result with markdown parser
                            const resultDiv = document.getElementById('resultContent');

                            // Create script to include marked.js for markdown parsing
                            if (!window.marked) {
                                const script = document.createElement('script');
                                script.src = 'https://cdnjs.cloudflare.com/ajax/libs/marked/4.0.0/marked.min.js';
                                document.head.appendChild(script);

                                script.onload = function() {
                                    // Use marked.js to render markdown
                                    resultDiv.innerHTML = marked.parse(data.result);
                                };
                            } else {
                                resultDiv.innerHTML = marked.parse(data.result);
                            }
                        } else if (data.status === 'failed') {
                            clearInterval(interval);
                            document.getElementById('loader').style.display = 'none';
                            document.getElementById('analyzeBtn').disabled = false;
                            document.getElementById('analyzeBtn').textContent = 'Analyze Transcript';
                            document.getElementById('resultContent').textContent = 'Error: ' + data.error;
                        }
                    })
                    .catch(error => {
                        console.error("Error during status check:", error);
                        document.getElementById('status').textContent = 'Error checking status: ' + error.message;
                        document.getElementById('loader').style.display = 'none';
                        clearInterval(interval);
                    });
                }, 10000); // Check every 10 seconds to avoid overloading the server
            }
        </script>
    </body>
    </html>
    """)


@app.post("/analyze/")
async def analyze_transcript(background_tasks: BackgroundTasks, transcript_file: UploadFile = File(...)):
    try:
        print(f"Received file: {transcript_file.filename}, Content-Type: {transcript_file.content_type}")

        # Validate file is a PDF
        if not transcript_file.content_type.endswith('/pdf'):
            raise HTTPException(
                status_code=400,
                detail=f"File must be a PDF. Received: {transcript_file.content_type}"
            )

        # Generate a unique task ID
        task_id = str(uuid.uuid4())

        # Save uploaded PDF to disk with unique name
        file_path = f"uploads/{task_id}_{transcript_file.filename}"
        with open(file_path, "wb") as f:
            shutil.copyfileobj(transcript_file.file, f)

        print(f"Saved PDF to disk: {file_path}")

        # Initialize task status immediately
        task_statuses[task_id] = {"status": "queued", "result": None}

        # Start the background task
        background_tasks.add_task(process_transcript, task_id, file_path)

        return JSONResponse(content={
            "message": "PDF uploaded successfully. Processing started.",
            "task_id": task_id,
            "file_info": {
                "filename": transcript_file.filename,
                "content_type": transcript_file.content_type,
                "saved_as": file_path
            }
        })

    except HTTPException as he:
        # Re-raise HTTP exceptions
        raise he
    except Exception as e:
        print(f"Error during upload: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a transcript analysis task"""
    if task_id not in task_statuses:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    try:
        # Return a copy of the task status to avoid potential issues with shared mutable state
        status_data = task_statuses[task_id].copy()
        return JSONResponse(content=status_data)
    except Exception as e:
        print(f"Error getting task status: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving task status: {str(e)}"
        )


if __name__ == "__main__":

    uvicorn.run(app, host="127.0.0.1", port=8000)