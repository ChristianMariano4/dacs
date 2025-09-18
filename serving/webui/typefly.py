import json
import queue
import sys, os
import asyncio
import io, time
from fastapi.responses import JSONResponse
import gradio as gr
from flask import Flask, Response, jsonify
from flask_cors import CORS
from threading import Thread
import argparse

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(PARENT_DIR)
from controller.context_map.graph_manager import GraphManager
from controller.llm.llm_controller import LLMController
from controller.utils import print_t
from controller.llm.llm_wrapper import GPT4, LLAMA3
from controller.abs.robot_wrapper import RobotType

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class TypeFly:
    def __init__(self, robot_type, use_http=False):
         # create a cache folder
        self.cache_folder = os.path.join(CURRENT_DIR, 'cache')
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)
        self.message_queue = queue.Queue()
        self.message_queue.put(self.cache_folder)
        self.user_answer_queue = queue.Queue()
        self.user_question_answer = [] # list of last question-answer pair between LLM and user
        self.llm_controller = LLMController(robot_type, use_http, self.message_queue, self.user_answer_queue)

        # Convert the JSON to a string
        init_graph_path = "controller/assets/tello/memory/graph.txt"

        # Initialize GraphManager with the sample data
        self.graph_manager = GraphManager(
            llm_controller=self.llm_controller, 
            init_graph_json=init_graph_path,
            start_region="kitchen_area",  # Start in kitchen
            start_coords=(200.0, 200.0)
        )
        # self.graph_manager = GraphManager(llm_controller=self.llm_controller, init_graph_json=)
        self.llm_controller.set_graph_manager(self.graph_manager)
        self.system_stop = False
        self.ui = gr.Blocks(title="TypeFly")
        self.asyncio_loop = asyncio.get_event_loop()
        self.use_llama3 = False
        # Graph log file path
        self.graph_log_path = os.path.join("graph_logs", "graph_history.jsonl")
        default_sentences = [
            "Find an apple",
            "Come back to region_0",
            "Where is a banana?",
            "Find a banana",
        ]
        with self.ui:
            gr.HTML(open(os.path.join(CURRENT_DIR, 'header.html'), 'r').read())
            # Create tabs for different views
            with gr.Tabs():
                with gr.TabItem("🤖 Robot Control"):
                    gr.HTML(open(os.path.join(CURRENT_DIR, 'drone-pov.html'), 'r').read())
                    gr.ChatInterface(self.process_message, fill_height=False, examples=default_sentences).queue()
                    
                with gr.TabItem("📊 Graph Visualization"):
                    self.setup_graph_tab()
                
                with gr.TabItem("⚙️ Settings"):
                    self.setup_settings_tab()
            # TODO: Add checkbox to switch between llama3 and gpt4
            # gr.Checkbox(label='Use llama3', value=False).select(self.checkbox_llama3)

    def setup_graph_tab(self):
        """Setup the graph visualization tab"""
        with gr.Column():
            gr.Markdown("## 🌐 Dynamic Object Detection Graph")
            gr.Markdown("Visualize the evolution of detected objects and their relationships over time.")
            
            # Graph visualization iframe
            graph_html = self.get_graph_html()
            self.graph_component = gr.HTML(graph_html)
            
            # Controls
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh Graph", variant="primary")
                clear_btn = gr.Button("🗑️ Clear Graph Data", variant="secondary")
                auto_refresh_btn = gr.Button("🔁 Auto Refresh", variant="secondary")
                
            # Status
            self.graph_status = gr.Textbox(
                label="Graph Status", 
                value="Ready to display graph data...",
                interactive=False
            )
            
            # Event handlers
            refresh_btn.click(self.refresh_graph, outputs=[self.graph_component, self.graph_status])
            clear_btn.click(self.clear_graph_data, outputs=[self.graph_status])
            auto_refresh_btn.click(self.toggle_auto_refresh, outputs=[self.graph_status])

    def setup_settings_tab(self):
        """Setup the Settings tab with flyzone generation and feedback."""
        with gr.Column():
            gr.Markdown("## 🌐 User Settings")
            gr.Markdown("Personalize your experience by adjusting options to suit your preferences.")

            # --- User name section ---
            username = gr.Textbox(
                label="Enter your name",
                placeholder="Type your name here...",
                lines=1
            )
            self.llm_controller.set_user_name(username)
            
            # --- Flyzone generation section ---
            gr.Markdown("### ✈️ Generate Flyzone")
            flyzone_prompt = gr.Textbox(
                label="Enter prompt to generate a flyzone",
                placeholder="Describe the area and shape of the flyzone...",
                lines=2
            )
            generate_btn = gr.Button("🚀 Generate Flyzone")

            # --- Feedback and result section ---
            status_output = gr.Label(label="Request Status")
            flyzone_image = gr.Image(
                label="Generated Flyzone Preview",
                type="filepath",
                visible=True
            )

            # Absolute path for flyzone image
            flyzone_img_path = os.path.abspath("controller/assets/tello/flyzone/flyzone_plot.png")

            # --- Handler function ---
            def handle_flyzone_request(instruction: str):
                yield "⏳ Generating flyzone, please wait...", None

                try:
                    # Call controller to request flyzone
                    self.llm_controller.get_flyzone_manager().request_new_flyzone(instruction=instruction)

                    # Show image if it exists
                    if os.path.exists(flyzone_img_path):
                        yield "✅ Flyzone generated successfully!", flyzone_img_path
                    else:
                        yield "⚠️ Flyzone generated, but no image found.", None

                except Exception as e:
                    yield f"❌ Failed to generate flyzone: {str(e)}", None

            # --- Connect UI events ---
            generate_btn.click(
                fn=handle_flyzone_request,
                inputs=flyzone_prompt,
                outputs=[status_output, flyzone_image]
            )

    def get_graph_html(self):
        """Generate the HTML for the graph visualization"""
        return """
        <div style="width: 100%; height: 750px; border: 1px solid #ddd; border-radius: 8px; overflow: hidden;">
            <iframe src="http://localhost:50000/graph" 
                    style="width: 100%; height: 100%; border: none;"
                    sandbox="allow-scripts allow-same-origin">
            </iframe>
        </div>
        """
    
    def refresh_graph(self):
        """Refresh the graph visualization"""
        try:
            # Get current graph state from graph manager
            current_graph = self.graph_manager.get_graph()
            if current_graph:
                status = "Graph refreshed! Displaying current graph state."
            else:
                status = "No graph data available. Start robot operations to generate data."
            
            return self.get_graph_html(), status
        except Exception as e:
            return self.get_graph_html(), f"Error refreshing graph: {str(e)}"

    def clear_graph_data(self):
        """Clear the graph log data"""
        try:
            if os.path.exists(self.graph_log_path):
                os.remove(self.graph_log_path)
                return "Graph data cleared successfully!"
            else:
                return "No graph data to clear."
        except Exception as e:
            return f"Error clearing graph data: {str(e)}"

    def toggle_auto_refresh(self):
        """Toggle auto-refresh mode"""
        return "Auto-refresh toggled! Graph will update automatically as new data arrives."
    
    def checkbox_llama3(self):
        self.use_llama3 = not self.use_llama3
        if self.use_llama3:
            print_t(f"Switch to llama3")
            self.llm_controller.planner.set_model(LLAMA3)
        else:
            print_t(f"Switch to gpt4")
            self.llm_controller.planner.set_model(GPT4)

    def process_message(self, message, history):
        print_t(f"[S] Receiving task description: {message}")
        if message == "exit":
            self.llm_controller.stop_controller()
            self.system_stop = True
            yield "Shutting down..."
        elif len(message) == 0:
            return "[WARNING] Empty command!]"
        elif len(self.user_question_answer) == 1: # the message inserted by user is the answer of previous question made by LLM
            print_t(f"[DEBUG] Treating as answer to: {self.user_question_answer[0]}")
            self.user_question_answer.append(message)
            temp = self.user_question_answer.copy()
            self.user_answer_queue.put(temp) # put in shared queue the pair to pass to llm_controller
            self.user_question_answer = []
            yield "Answer sent. Wait a second and I will be here again"

            # Continue processing messages from the queue
            complete_response = 'Answer sent\n'
            while True:
                msg = self.message_queue.get()
                if isinstance(msg, tuple):
                    history.append((None, msg))
                elif isinstance(msg, str):
                    if msg == 'end':
                        return complete_response + "\nCommand Complete!"
                    
                    if msg.startswith('[LOG]') or msg.startswith('[Q]'):
                        complete_response += '\n'
                    if msg.startswith('[Q]'):
                        self.user_question_answer.append(msg)

                    if msg.endswith('\\\\'):
                        complete_response += msg.rstrip('\\\\')
                    else:
                        complete_response += msg + '\n'
                yield complete_response
        else:
            task_thread = Thread(target=self.llm_controller.execute_task_description, args=(message,))
            task_thread.start()
            complete_response = ''
            while True:
                msg = self.message_queue.get()
                if isinstance(msg, tuple):
                    history.append((None, msg))
                elif isinstance(msg, str):
                    if msg == 'end':
                        return "Command Complete!"
                    
                    if msg.startswith('[LOG]') or msg.startswith('[Q]'):
                        complete_response += '\n'
                    if msg.startswith('[Q]'):
                        self.user_question_answer.append(msg)

                    if msg.endswith('\\\\'):
                        complete_response += msg.rstrip('\\\\')
                    else:
                        complete_response += msg + '\n'
                yield complete_response

    def generate_mjpeg_stream(self):
        while True:
            if self.system_stop:
                break
            frame = self.llm_controller.get_latest_frame(True)
            if frame is None:
                continue
            buf = io.BytesIO()
            frame.save(buf, format='JPEG')
            buf.seek(0)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.read() + b'\r\n')
            time.sleep(1.0 / 30.0)

    def setup_graph_server(self, app):
        """Setup the graph visualization server endpoint"""
        @app.route('/graph')
        def graph_page():
            # Return the enhanced graph visualization HTML
            return '''
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>TypeFly Graph Visualization</title>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
                <style>
                    body {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                        color: white;
                    }
                    
                    .container {
                        max-width: 1200px;
                        margin: 0 auto;
                        background: rgba(255, 255, 255, 0.1);
                        backdrop-filter: blur(10px);
                        border-radius: 20px;
                        padding: 30px;
                        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
                    }
                    
                    h1 {
                        text-align: center;
                        margin-bottom: 30px;
                        font-size: 2.5em;
                        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
                    }
                    
                    .controls {
                        display: flex;
                        gap: 15px;
                        margin-bottom: 20px;
                        flex-wrap: wrap;
                        align-items: center;
                    }
                    
                    button {
                        padding: 12px 24px;
                        border: none;
                        border-radius: 25px;
                        background: linear-gradient(45deg, #ff6b6b, #ee5a24);
                        color: white;
                        font-weight: bold;
                        cursor: pointer;
                        transition: all 0.3s ease;
                        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                    }
                    
                    button:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
                    }
                    
                    button:disabled {
                        opacity: 0.5;
                        cursor: not-allowed;
                        transform: none;
                    }
                    
                    .graph-container {
                        background: rgba(0, 0, 0, 0.1);
                        border-radius: 15px;
                        padding: 20px;
                        margin-top: 20px;
                        box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.2);
                        overflow: auto; /* adds scrollbars if graph is bigger */
                    }
                    
                    .status {
                        padding: 10px 20px;
                        background: rgba(255, 255, 255, 0.1);
                        border-radius: 10px;
                        margin: 10px 0;
                        font-weight: bold;
                    }
                    
                    .legend {
                        display: flex;
                        gap: 20px;
                        margin: 10px 0;
                        flex-wrap: wrap;
                    }
                    
                    .legend-item {
                        display: flex;
                        align-items: center;
                        gap: 8px;
                        padding: 8px 15px;
                        background: rgba(255, 255, 255, 0.1);
                        border-radius: 20px;
                    }
                    
                    .legend-color {
                        width: 16px;
                        height: 16px;
                        border-radius: 50%;
                    }
                    
                    .node {
                        cursor: pointer;
                        transition: all 0.3s ease;
                    }
                    
                    .node:hover {
                        stroke-width: 3px;
                        stroke: #fff;
                    }
                    
                    .link {
                        stroke: rgba(255, 255, 255, 0.6);
                        stroke-width: 2px;
                        transition: all 0.3s ease;
                    }
                    
                    .link:hover {
                        stroke: #fff;
                        stroke-width: 3px;
                    }
                    
                    .node-label {
                        font-size: 11px;
                        font-weight: bold;
                        text-anchor: middle;
                        fill: white;
                        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
                        pointer-events: none;
                    }
                    
                    .tooltip {
                        position: absolute;
                        background: rgba(0, 0, 0, 0.9);
                        color: white;
                        padding: 10px;
                        border-radius: 8px;
                        font-size: 12px;
                        pointer-events: none;
                        opacity: 0;
                        transition: opacity 0.3s ease;
                        z-index: 1000;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🤖 TypeFly Graph Visualization</h1>
                    
                    <div class="controls">
                        <button id="loadBtn">📁 Load Current Graph</button>
                        <button id="refreshBtn">🔄 Refresh</button>
                        <button id="autoRefreshBtn">🔁 Auto Refresh</button>
                    </div>
                    
                    <div class="status" id="status">Ready to load graph data...</div>
                    
                    <div class="legend">
                        <div class="legend-item">
                            <div class="legend-color" style="background: #ff6b6b;"></div>
                            <span>Objects</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #4ecdc4;"></div>
                            <span>Regions</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #ffd700;"></div>
                            <span>Current Location</span>
                        </div>
                    </div>
                    
                    <div class="graph-container">
                        <svg id="graph" width="100%" height="600"></svg>
                    </div>
                    
                    <div class="tooltip" id="tooltip"></div>
                </div>

                <script>
                    class TypeFlyGraphVisualizer {
                        constructor() {
                            this.svg = d3.select("#graph");
                            this.width = 1000;
                            this.height = 600;
                            this.currentGraph = null;
                            this.autoRefresh = false;
                            this.autoRefreshInterval = null;
                            
                            this.svg.attr("viewBox", `0 0 ${this.width} ${this.height}`);
                            
                            this.setupSimulation();
                            this.setupEventListeners();
                            this.loadData(); // Auto-load on start
                        }
                        
                        setupSimulation() {
                            this.simulation = d3.forceSimulation()
                                .force("link", d3.forceLink().id(d => d.id).distance(150))
                                .force("charge", d3.forceManyBody().strength(-400))
                                .force("center", d3.forceCenter(this.width / 2, this.height / 2))
                                .force("collision", d3.forceCollide().radius(40));
                            
                            this.linkGroup = this.svg.append("g").attr("class", "links");
                            this.nodeGroup = this.svg.append("g").attr("class", "nodes");
                            this.labelGroup = this.svg.append("g").attr("class", "labels");
                        }
                        
                        setupEventListeners() {
                            document.getElementById('loadBtn').addEventListener('click', () => this.loadData());
                            document.getElementById('refreshBtn').addEventListener('click', () => this.loadData());
                            document.getElementById('autoRefreshBtn').addEventListener('click', () => this.toggleAutoRefresh());
                        }
                        
                        async loadData() {
                            try {
                                const response = await fetch('/graph-data');
                                const result = await response.json();
                                
                                if (result.success && result.data) {
                                    this.currentGraph = result.data;
                                    this.updateVisualization();
                                    this.updateStatus('Current graph state loaded successfully');
                                } else {
                                    this.updateStatus(result.message || 'No graph data available');
                                }
                            } catch (error) {
                                this.updateStatus('Error loading data: ' + error.message);
                            }
                        }
                        
                        toggleAutoRefresh() {
                            this.autoRefresh = !this.autoRefresh;
                            if (this.autoRefresh) {
                                this.updateStatus('Auto-refresh enabled - checking for updates every 3 seconds');
                                this.autoRefreshInterval = setInterval(() => {
                                    this.loadData();
                                }, 3000);
                                document.getElementById('autoRefreshBtn').textContent = '⏹️ Stop Auto Refresh';
                            } else {
                                this.updateStatus('Auto-refresh disabled');
                                if (this.autoRefreshInterval) {
                                    clearInterval(this.autoRefreshInterval);
                                    this.autoRefreshInterval = null;
                                }
                                document.getElementById('autoRefreshBtn').textContent = '🔁 Auto Refresh';
                            }
                        }
                        
                        parseGraphData(graphStr) {
                            try {
                                const graph = JSON.parse(graphStr);
                                
                                const nodes = [];
                                const links = [];
                                
                                // Add objects as nodes
                                if (graph.objects && Array.isArray(graph.objects)) {
                                    graph.objects.forEach(obj => {
                                        let coords = null;
                                        if (obj.coords) {
                                            try {
                                                coords = typeof obj.coords === 'string' ? JSON.parse(obj.coords) : obj.coords;
                                            } catch (e) {
                                                console.warn('Invalid coords for object:', obj.name);
                                            }
                                        }
                                        nodes.push({
                                            id: obj.name,
                                            type: 'object',
                                            coords: coords
                                        });
                                    });
                                }
                                
                                // Add regions as nodes
                                if (graph.regions && Array.isArray(graph.regions)) {
                                    graph.regions.forEach(region => {
                                        let coords = null;
                                        if (region.coords) {
                                            try {
                                                coords = typeof region.coords === 'string' ? JSON.parse(region.coords) : region.coords;
                                            } catch (e) {
                                                console.warn('Invalid coords for region:', region.name);
                                            }
                                        }
                                        nodes.push({
                                            id: region.name,
                                            type: 'region',
                                            coords: coords,
                                            isCurrent: region.name === graph.current_location
                                        });
                                    });
                                }
                                
                                // Add object connections
                                if (graph.object_connections && Array.isArray(graph.object_connections)) {
                                    graph.object_connections.forEach(conn => {
                                        if (Array.isArray(conn) && conn.length >= 2) {
                                            links.push({
                                                source: conn[0],
                                                target: conn[1],
                                                type: 'object_connection'
                                            });
                                        }
                                    });
                                }
                                
                                // Add region connections
                                if (graph.region_connections && Array.isArray(graph.region_connections)) {
                                    graph.region_connections.forEach(conn => {
                                        if (Array.isArray(conn) && conn.length >= 2) {
                                            links.push({
                                                source: conn[0],
                                                target: conn[1],
                                                type: 'region_connection'
                                            });
                                        }
                                    });
                                }
                                
                                return { nodes, links };
                            } catch (e) {
                                console.error('Error parsing graph data:', e);
                                console.error('Graph string was:', graphStr);
                                return { nodes: [], links: [] };
                            }
                        }
                        
                        getNodeColor(node) {
                            if (node.isCurrent) {
                                return '#ffd700'; // Gold for current location
                            }
                            switch (node.type) {
                                case 'object': return '#ff6b6b';
                                case 'region': return '#4ecdc4';
                                default: return '#45b7d1';
                            }
                        }
                        
                        getLinkColor(link) {
                            switch (link.type) {
                                case 'object_connection': return 'rgba(255, 107, 107, 0.6)';
                                case 'region_connection': return 'rgba(78, 205, 196, 0.6)';
                                default: return 'rgba(255, 255, 255, 0.6)';
                            }
                        }
                        
                        updateVisualization() {
                            if (!this.currentGraph) {
                                this.updateStatus('No graph data to display');
                                return;
                            }
                            
                            const graphData = this.parseGraphData(this.currentGraph);
                            
                            if (graphData.nodes.length === 0) {
                                this.updateStatus('Graph has no nodes to display');
                                return;
                            }
                            
                            // Clear existing elements
                            this.linkGroup.selectAll("*").remove();
                            this.nodeGroup.selectAll("*").remove();
                            this.labelGroup.selectAll("*").remove();
                            
                            // Add links
                            const links = this.linkGroup.selectAll("line")
                                .data(graphData.links)
                                .enter()
                                .append("line")
                                .attr("class", "link")
                                .style("stroke", d => this.getLinkColor(d))
                                .style("stroke-width", d => d.type === 'region_connection' ? 3 : 2)
                                .style("opacity", 0.8);
                            
                            // Add nodes
                            const nodes = this.nodeGroup.selectAll("circle")
                                .data(graphData.nodes)
                                .enter()
                                .append("circle")
                                .attr("class", "node")
                                .attr("r", d => d.type === 'region' ? 25 : 15)
                                .style("fill", d => this.getNodeColor(d))
                                .style("stroke", d => d.isCurrent ? '#fff' : 'none')
                                .style("stroke-width", d => d.isCurrent ? 4 : 0)
                                .on("mouseover", (event, d) => {
                                    const tooltip = document.getElementById('tooltip');
                                    tooltip.innerHTML = `
                                        <strong>${d.id}</strong><br>
                                        Type: ${d.type}<br>
                                        ${d.coords ? `Coords: [${d.coords[0]?.toFixed(1) || 'N/A'}, ${d.coords[1]?.toFixed(1) || 'N/A'}]` : 'No coordinates'}
                                        ${d.isCurrent ? '<br><em>📍 Current Location</em>' : ''}
                                    `;
                                    tooltip.style.opacity = 1;
                                    tooltip.style.left = (event.pageX + 10) + 'px';
                                    tooltip.style.top = (event.pageY - 10) + 'px';
                                })
                                .on("mouseout", () => {
                                    document.getElementById('tooltip').style.opacity = 0;
                                })
                            
                            // Add labels
                            const labels = this.labelGroup.selectAll("text")
                                .data(graphData.nodes)
                                .enter()
                                .append("text")
                                .attr("class", "node-label")
                                .text(d => d.id.length > 12 ? d.id.substring(0, 12) + '...' : d.id);
                            
                            // Update simulation
                            this.simulation.nodes(graphData.nodes);
                            this.simulation.force("link").links(graphData.links);
                            
                            this.simulation.on("tick", () => {
                                links
                                    .attr("x1", d => d.source.x)
                                    .attr("y1", d => d.source.y)
                                    .attr("x2", d => d.target.x)
                                    .attr("y2", d => d.target.y);
                                
                                nodes
                                    .attr("cx", d => d.x)
                                    .attr("cy", d => d.y);
                                
                                labels
                                    .attr("x", d => d.x)
                                    .attr("y", d => d.y + 4);
                            });
                            
                            this.simulation.alpha(1).restart();
                            
                            const objectCount = graphData.nodes.filter(n => n.type === 'object').length;
                            const regionCount = graphData.nodes.filter(n => n.type === 'region').length;
                            this.updateStatus(`Graph loaded - Objects: ${objectCount}, Regions: ${regionCount}, Links: ${graphData.links.length}`);
                        }
                        
                        updateStatus(message) {
                            document.getElementById('status').textContent = message;
                        }
                    }
                    
                    // Initialize the TypeFly graph visualizer
                    const visualizer = new TypeFlyGraphVisualizer();
                </script>
            </body>
            </html>
            '''
        
        @app.route('/graph-data')
        def graph_data():
            """Serve the current graph data as JSON"""
            try:
                # Get current graph state from graph manager
                graph_json_str = self.graph_manager.get_graph()
                
                if not graph_json_str:
                    return jsonify({
                        "success": False, 
                        "data": None, 
                        "message": "No graph data available - start robot operations to generate data"
                    })

                # Parse the JSON string to validate it
                try:
                    graph_dict = json.loads(graph_json_str)
                except json.JSONDecodeError as e:
                    return jsonify({
                        "success": False, 
                        "data": None, 
                        "message": f"Invalid graph JSON format: {str(e)}"
                    })

                return jsonify({
                    "success": True, 
                    "data": graph_json_str,  # Send as string for JavaScript to parse
                    "message": "Current graph state retrieved successfully"
                })

            except Exception as e:
                return jsonify({
                    "success": False, 
                    "data": None, 
                    "message": f"Error retrieving graph: {str(e)}"
                })

    def run(self):
        asyncio_thread = Thread(target=self.asyncio_loop.run_forever)
        asyncio_thread.start()

        self.llm_controller.start_robot()
        llmc_thread = Thread(target=self.llm_controller.capture_loop, args=(self.asyncio_loop,))
        llmc_thread.start()

        app = Flask(__name__)
        CORS(app)  # allow all origins

        @app.route('/drone-pov/')
        def video_feed():
            return Response(self.generate_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')
        
        # Add graph visualization endpoints
        self.setup_graph_server(app)

        flask_thread = Thread(target=app.run, kwargs={'host': 'localhost', 'port': 50000, 'debug': False, 'use_reloader': False})
        flask_thread.start()
        
        self.ui.launch(show_api=False, server_port=50001, prevent_thread_lock=True)
        
        while True:
            time.sleep(1)
            if self.system_stop:
                break

        llmc_thread.join()
        asyncio_thread.join()
        self.llm_controller.stop_robot()

        # clean self.cache_folder
        for file in os.listdir(self.cache_folder):
            os.remove(os.path.join(self.cache_folder, file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_virtual_robot', action='store_true')
    parser.add_argument('--use_http', action='store_true')
    parser.add_argument('--gear', action='store_true')
    parser.add_argument('--crazyflie', action='store_true')

    args = parser.parse_args()
    robot_type = RobotType.TELLO
    if args.use_virtual_robot:
        robot_type = RobotType.VIRTUAL
    elif args.gear:
        robot_type = RobotType.GEAR
    elif args.crazyflie:
        robot_type = RobotType.CRAZYFLIE
    typefly = TypeFly(robot_type, use_http=args.use_http)
    typefly.run()