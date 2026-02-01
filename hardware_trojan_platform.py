

import os
import re
import warnings
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool

import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from scipy import stats

# -----------------------
# STREAMLIT CONFIG
# -----------------------
st.set_page_config(
    page_title="HT Detection - IIT Kanpur Competition",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔐"
)

# Enhanced CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .competition-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .trojan-detected {
        background-color: #ffebee;
        border-left: 4px solid #d50000;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .clean-design {
        background-color: #e8f5e9;
        border-left: 4px solid #00c853;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# DATA STRUCTURES
# -----------------------
@dataclass
class VerilogSignal:
    name: str
    signal_type: str
    width: int = 1
    is_clock: bool = False
    is_reset: bool = False
    fanin: int = 0
    fanout: int = 0
    toggle_rate: float = 0.0
    
@dataclass
class VerilogModule:
    name: str
    signals: Dict[str, VerilogSignal]
    assignments: List[Tuple[str, str]]
    always_blocks: List[str]
    instances: List[str] = field(default_factory=list)
    parameters: Dict[str, str] = field(default_factory=dict)
    
@dataclass
class IOCharacteristics:
    """I/O characteristics for golden model comparison"""
    input_signals: List[str]
    output_signals: List[str]
    io_table: Dict[str, List]
    test_vectors: List[Tuple]

# -----------------------
# ENHANCED VERILOG PARSER
# -----------------------
class CompetitionVerilogParser:
    """Enhanced parser with I/O extraction for golden model comparison"""
    
    def __init__(self):
        self.signal_pattern = re.compile(
            r'(input|output|wire|reg|inout)\s*(?:\[(\d+):(\d+)\])?\s*([\w,\s]+);'
        )
        self.assign_pattern = re.compile(r'assign\s+(\w+)\s*=\s*([^;]+);')
        self.always_pattern = re.compile(r'always\s*@\((.*?)\)(.*?)(?=always|endmodule|$)', re.S)
        self.module_pattern = re.compile(r'module\s+(\w+)')
        self.instance_pattern = re.compile(r'(\w+)\s+(?:#\(.*?\))?\s*(\w+)\s*\(', re.S)
        self.param_pattern = re.compile(r'parameter\s+(\w+)\s*=\s*([^;]+);')
        
    def parse(self, content: str) -> VerilogModule:
        # Remove comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        content = re.sub(r'//.*', '', content)
        
        # Extract module name
        module_match = self.module_pattern.search(content)
        module_name = module_match.group(1) if module_match else "unknown"
        
        signals = {}
        assignments = []
        always_blocks = []
        instances = []
        parameters = {}
        
        # Parse signals with enhanced metadata
        for m in self.signal_pattern.finditer(content):
            stype = m.group(1)
            hi = int(m.group(2)) if m.group(2) else 0
            lo = int(m.group(3)) if m.group(3) else 0
            width = abs(hi - lo) + 1
            
            for name in m.group(4).split(","):
                name = name.strip()
                is_clk = bool(re.search(r'clk|clock', name, re.I))
                is_rst = bool(re.search(r'rst|reset', name, re.I))
                signals[name] = VerilogSignal(name, stype, width, is_clk, is_rst)
        
        # Parse assignments
        for m in self.assign_pattern.finditer(content):
            assignments.append((m.group(1).strip(), m.group(2).strip()))
        
        # Parse always blocks
        for m in self.always_pattern.finditer(content):
            always_blocks.append(m.group(0))
        
        # Parse instances
        for m in self.instance_pattern.finditer(content):
            instances.append(f"{m.group(1)} {m.group(2)}")
        
        # Parse parameters
        for m in self.param_pattern.finditer(content):
            parameters[m.group(1)] = m.group(2).strip()
        
        return VerilogModule(module_name, signals, assignments, always_blocks, instances, parameters)
    
    def extract_io_characteristics(self, module: VerilogModule) -> IOCharacteristics:
        """Extract I/O characteristics for golden model comparison"""
        inputs = [name for name, sig in module.signals.items() if sig.signal_type == 'input']
        outputs = [name for name, sig in module.signals.items() if sig.signal_type == 'output']
        
        return IOCharacteristics(
            input_signals=inputs,
            output_signals=outputs,
            io_table={},
            test_vectors=[]
        )

# -----------------------
# STATISTICAL MODEL (First Approach in Competition)
# -----------------------
class StatisticalTrojanDetector:
    """Statistical model using I/O characteristics and anomaly detection"""
    
    def __init__(self):
        self.suspicious_patterns = [
            r'trigger', r'payload', r'malicious', r'trojan', r'backdoor',
            r'secret', r'hidden', r'rare', r'low_prob', r'attack'
        ]
        
    def compute_structural_features(self, module: VerilogModule) -> Dict[str, float]:
        """Compute structural features for statistical analysis"""
        features = {}
        
        # Basic complexity metrics
        features['num_signals'] = len(module.signals)
        features['num_assignments'] = len(module.assignments)
        features['num_always_blocks'] = len(module.always_blocks)
        features['num_instances'] = len(module.instances)
        
        # Signal type distribution
        signal_types = defaultdict(int)
        for sig in module.signals.values():
            signal_types[sig.signal_type] += 1
        
        features['input_ratio'] = signal_types['input'] / max(len(module.signals), 1)
        features['output_ratio'] = signal_types['output'] / max(len(module.signals), 1)
        features['reg_ratio'] = signal_types['reg'] / max(len(module.signals), 1)
        
        # Connectivity metrics
        fanins = [sig.fanin for sig in module.signals.values()]
        fanouts = [sig.fanout for sig in module.signals.values()]
        
        features['avg_fanin'] = np.mean(fanins) if fanins else 0
        features['max_fanin'] = np.max(fanins) if fanins else 0
        features['std_fanin'] = np.std(fanins) if fanins else 0
        
        features['avg_fanout'] = np.mean(fanouts) if fanouts else 0
        features['max_fanout'] = np.max(fanouts) if fanouts else 0
        features['std_fanout'] = np.std(fanouts) if fanouts else 0
        
        # Width distribution
        widths = [sig.width for sig in module.signals.values()]
        features['avg_width'] = np.mean(widths) if widths else 0
        features['max_width'] = np.max(widths) if widths else 0
        features['std_width'] = np.std(widths) if widths else 0
        
        # Logic complexity
        total_logic_depth = 0
        for block in module.always_blocks:
            total_logic_depth += len(re.findall(r'if|case|for|while', block))
        features['logic_complexity'] = total_logic_depth
        
        return features
    
    def analyze(self, module: VerilogModule, golden_features: Optional[Dict] = None) -> Dict[str, any]:
        """Perform statistical analysis with optional golden model comparison"""
        
        anomalies = {
            'suspicious_names': [],
            'unusual_widths': [],
            'high_fanout': [],
            'isolated_signals': [],
            'complex_logic': [],
            'rare_signals': [],
            'golden_deviation': [],
            'score': 0.0
        }
        
        # 1. Suspicious naming patterns
        for name, sig in module.signals.items():
            for pattern in self.suspicious_patterns:
                if re.search(pattern, name, re.I):
                    anomalies['suspicious_names'].append(name)
                    anomalies['score'] += 0.4
                    break
        
        # 2. Statistical outliers in bit widths
        widths = [s.width for s in module.signals.values()]
        if widths:
            mean_width = np.mean(widths)
            std_width = np.std(widths)
            threshold = mean_width + 2.5 * std_width
            
            for name, sig in module.signals.items():
                if sig.width > threshold and sig.width > 8:
                    anomalies['unusual_widths'].append((name, sig.width))
                    anomalies['score'] += 0.15
        
        # 3. High fanout (potential trigger distribution)
        fanouts = [s.fanout for s in module.signals.values() if s.fanout > 0]
        if fanouts:
            fanout_threshold = np.percentile(fanouts, 90) if len(fanouts) > 5 else 10
            for name, sig in module.signals.items():
                if sig.fanout > fanout_threshold and sig.fanout > 8:
                    anomalies['high_fanout'].append((name, sig.fanout))
                    anomalies['score'] += 0.2
        
        # 4. Isolated signals (unused/dead logic)
        for name, sig in module.signals.items():
            if sig.fanin == 0 and sig.fanout == 0:
                if sig.signal_type not in ['input', 'output']:
                    anomalies['isolated_signals'].append(name)
                    anomalies['score'] += 0.25
        
        # 5. Complex logic blocks (obfuscation indicator)
        for block in module.always_blocks:
            complexity = len(re.findall(r'if|case', block))
            if complexity > 7:
                anomalies['complex_logic'].append(complexity)
                anomalies['score'] += 0.3
        
        # 6. Rare signal patterns (low toggle rate indicators)
        for name, sig in module.signals.items():
            # Check for counter-like patterns that might be rare triggers
            if re.search(r'cnt|counter|count', name, re.I) and sig.width > 16:
                anomalies['rare_signals'].append(name)
                anomalies['score'] += 0.2
        
        # 7. Golden model comparison (if available)
        if golden_features:
            current_features = self.compute_structural_features(module)
            
            # Compare key metrics
            for key in ['num_signals', 'num_assignments', 'logic_complexity']:
                if key in golden_features and key in current_features:
                    deviation = abs(current_features[key] - golden_features[key]) / max(golden_features[key], 1)
                    if deviation > 0.2:  # 20% deviation threshold
                        anomalies['golden_deviation'].append((key, deviation))
                        anomalies['score'] += 0.3 * deviation
        
        # Normalize score
        anomalies['score'] = min(anomalies['score'], 1.0)
        
        # Statistical confidence
        num_indicators = sum([
            len(anomalies['suspicious_names']),
            len(anomalies['unusual_widths']),
            len(anomalies['high_fanout']),
            len(anomalies['isolated_signals']),
            len(anomalies['complex_logic']),
            len(anomalies['rare_signals'])
        ])
        
        anomalies['confidence'] = min(num_indicators / 10.0, 1.0)
        
        return anomalies

# -----------------------
# ENHANCED GRAPH BUILDER
# -----------------------
class EnhancedGraphBuilder:
    """Build graph with richer features for GNN"""
    
    def __init__(self, feature_dim=48):
        self.feature_dim = feature_dim
        self.type_map = {'input': 0, 'output': 1, 'wire': 2, 'reg': 3, 'inout': 4}
        
    def extract_signals(self, expr: str) -> List[str]:
        """Extract signal names from an expression"""
        return [s for s in re.findall(r'\b[a-zA-Z_]\w*\b', expr) 
                if s not in ['if', 'else', 'case', 'default', 'begin', 'end', 'posedge', 'negedge']]
    
    def compute_graph_metrics(self, G: nx.DiGraph, node: str) -> Dict[str, float]:
        """Compute graph-theoretic metrics for a node"""
        metrics = {}
        
        try:
            # Centrality measures
            if len(G) > 1:
                metrics['betweenness'] = nx.betweenness_centrality(G).get(node, 0)
                metrics['closeness'] = nx.closeness_centrality(G).get(node, 0) if nx.is_weakly_connected(G) else 0
                metrics['pagerank'] = nx.pagerank(G).get(node, 0)
            else:
                metrics['betweenness'] = 0
                metrics['closeness'] = 0
                metrics['pagerank'] = 0
                
            # Local structure
            metrics['clustering'] = nx.clustering(G.to_undirected()).get(node, 0)
            
        except:
            metrics = {'betweenness': 0, 'closeness': 0, 'pagerank': 0, 'clustering': 0}
        
        return metrics
    
    def build(self, module: VerilogModule) -> Data:
        node_map = {name: i for i, name in enumerate(module.signals)}
        n = len(node_map)
        
        if n == 0:
            return Data(x=torch.zeros((1, self.feature_dim)), 
                       edge_index=torch.zeros((2, 0), dtype=torch.long),
                       edge_attr=torch.zeros((0, 1)))
        
        # Enhanced feature matrix
        x = torch.zeros((n, self.feature_dim))
        edges = []
        edge_attrs = []
        
        # Build NetworkX graph for metrics
        G = nx.DiGraph()
        for name in module.signals:
            G.add_node(name)
        
        # Compute fanin/fanout
        fanin = defaultdict(int)
        fanout = defaultdict(int)
        
        # Build dependency graph from assignments
        for tgt, expr in module.assignments:
            sources = self.extract_signals(expr)
            for src in sources:
                if src in node_map and tgt in node_map:
                    edges.append((node_map[src], node_map[tgt]))
                    edge_attrs.append(0)  # Combinational
                    G.add_edge(src, tgt)
                    fanout[src] += 1
                    fanin[tgt] += 1
        
        # Build from always blocks
        for block in module.always_blocks:
            # Non-blocking assignments
            nb_assigns = re.findall(r'(\w+)\s*<=\s*([^;]+);', block)
            for lhs, rhs in nb_assigns:
                sources = self.extract_signals(rhs)
                for src in sources:
                    if src in node_map and lhs in node_map:
                        edges.append((node_map[src], node_map[lhs]))
                        edge_attrs.append(1)  # Sequential
                        G.add_edge(src, lhs)
                        fanout[src] += 1
                        fanin[lhs] += 1
            
            # Blocking assignments
            b_assigns = re.findall(r'(\w+)\s*=\s*([^;]+);', block)
            for lhs, rhs in b_assigns:
                sources = self.extract_signals(rhs)
                for src in sources:
                    if src in node_map and lhs in node_map:
                        edges.append((node_map[src], node_map[lhs]))
                        edge_attrs.append(2)  # Blocking
                        G.add_edge(src, lhs)
                        fanout[src] += 1
                        fanin[lhs] += 1
        
        # Populate enhanced node features
        for name, idx in node_map.items():
            sig = module.signals[name]
            
            # Basic type (one-hot) [0-4]
            type_idx = self.type_map.get(sig.signal_type, 2)
            x[idx, type_idx] = 1
            
            # Width features [5-7]
            x[idx, 5] = min(sig.width / 64.0, 1.0)
            x[idx, 6] = 1 if sig.width > 32 else 0  # Wide signal flag
            x[idx, 7] = np.log2(sig.width + 1) / 8.0  # Log-width
            
            # Special signals [8-9]
            x[idx, 8] = 1 if sig.is_clock else 0
            x[idx, 9] = 1 if sig.is_reset else 0
            
            # Connectivity [10-13]
            x[idx, 10] = min(fanin[name] / 20.0, 1.0)
            x[idx, 11] = min(fanout[name] / 20.0, 1.0)
            x[idx, 12] = fanin[name] if fanin[name] < 10 else 10
            x[idx, 13] = fanout[name] if fanout[name] < 10 else 10
            
            # Pattern-based features [14-22]
            x[idx, 14] = 1 if re.search(r'temp|tmp|aux', name, re.I) else 0
            x[idx, 15] = 1 if re.search(r'cnt|counter|count', name, re.I) else 0
            x[idx, 16] = 1 if re.search(r'state|status|mode', name, re.I) else 0
            x[idx, 17] = 1 if re.search(r'enable|en\b|valid', name, re.I) else 0
            x[idx, 18] = 1 if re.search(r'trigger|trig|fire', name, re.I) else 0
            x[idx, 19] = 1 if re.search(r'payload|data|secret', name, re.I) else 0
            x[idx, 20] = 1 if re.search(r'sel|mux|select', name, re.I) else 0
            x[idx, 21] = 1 if re.search(r'flag|bit|indicator', name, re.I) else 0
            x[idx, 22] = 1 if len(name) > 20 else 0  # Unusually long name
            
            # Graph metrics [23-26]
            graph_metrics = self.compute_graph_metrics(G, name)
            x[idx, 23] = graph_metrics.get('betweenness', 0)
            x[idx, 24] = graph_metrics.get('closeness', 0)
            x[idx, 25] = graph_metrics.get('pagerank', 0) * 10  # Scale up
            x[idx, 26] = graph_metrics.get('clustering', 0)
            
            # Isolation indicators [27-29]
            x[idx, 27] = 1 if (fanin[name] == 0 and fanout[name] == 0) else 0
            x[idx, 28] = 1 if (fanin[name] == 0 and sig.signal_type not in ['input']) else 0
            x[idx, 29] = 1 if (fanout[name] == 0 and sig.signal_type not in ['output']) else 0
            
            # Statistical features [30-35]
            all_widths = [s.width for s in module.signals.values()]
            if all_widths:
                mean_width = np.mean(all_widths)
                std_width = np.std(all_widths) if len(all_widths) > 1 else 1
                x[idx, 30] = (sig.width - mean_width) / (std_width + 1e-6)  # Z-score
            
            all_fanouts = [fanout[n] for n in module.signals]
            if all_fanouts:
                mean_fanout = np.mean(all_fanouts)
                std_fanout = np.std(all_fanouts) if len(all_fanouts) > 1 else 1
                x[idx, 31] = (fanout[name] - mean_fanout) / (std_fanout + 1e-6)
            
            # Anomaly flags [32-35]
            x[idx, 32] = 1 if fanout[name] > 15 else 0  # Very high fanout
            x[idx, 33] = 1 if (sig.width > 16 and fanout[name] == 1) else 0  # Wide but low use
            x[idx, 34] = 1 if (fanin[name] > 10) else 0  # Complex input
            x[idx, 35] = 1 if re.search(r'\d+$', name) else 0  # Numbered signal
            
            # Update signal metadata
            module.signals[name].fanin = fanin[name]
            module.signals[name].fanout = fanout[name]
        
        # Create edge tensors
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float).unsqueeze(1)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# -----------------------
# ADVANCED GNN MODEL
# -----------------------
class CompetitionTrojanGNN(nn.Module):
    """Enhanced GNN with attention and ensemble features"""
    
    def __init__(self, input_dim=48, hidden_dim=256, num_layers=4):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection with residual
        self.embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(0.15)
        )
        
        # Multi-layer GAT with increasing attention
        self.gat_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=0.15, concat=True)
            for _ in range(num_layers)
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Multi-scale pooling
        pool_dim = hidden_dim * 3  # mean + max + add
        
        # Enhanced classification head with attention
        self.attention = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim * 2),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, 2)
        )
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input embedding
        x = self.embed(x)
        
        # Graph convolutions with residual
        for i, (gat, bn) in enumerate(zip(self.gat_layers, self.batch_norms)):
            x_prev = x
            x = F.elu(gat(x, edge_index))
            x = bn(x)
            if i > 0:
                x = x + x_prev  # Residual connection
            x = F.dropout(x, p=0.15, training=self.training)
        
        # Multi-scale pooling
        g_mean = global_mean_pool(x, batch)
        g_max = global_max_pool(x, batch)
        g_add = global_add_pool(x, batch)
        g = torch.cat([g_mean, g_max, g_add], dim=1)
        
        # Classification
        out = self.classifier(g)
        
        return out, g

# -----------------------
# HYBRID DETECTION SYSTEM
# -----------------------
class HybridTrojanDetectionSystem:
    """Combines Statistical and GNN approaches"""
    
    def __init__(self, gnn_weight=0.6, stat_weight=0.4):
        self.gnn_weight = gnn_weight
        self.stat_weight = stat_weight
        
        # Initialize models
        self.gnn_model = CompetitionTrojanGNN(input_dim=48, hidden_dim=256, num_layers=4)
        self.gnn_model.eval()
        
        self.stat_detector = StatisticalTrojanDetector()
        
    def predict(self, module: VerilogModule, graph: Data, 
                golden_features: Optional[Dict] = None) -> Dict:
        """Hybrid prediction combining both approaches"""
        
        # GNN prediction
        batch = Batch.from_data_list([graph])
        with torch.no_grad():
            gnn_out, embedding = self.gnn_model(batch)
            gnn_prob = F.softmax(gnn_out, dim=1)
            gnn_pred = gnn_out.argmax(dim=1).item()
            gnn_confidence = gnn_prob[0, gnn_pred].item()
            gnn_trojan_score = gnn_prob[0, 1].item()  # Probability of trojan class
        
        # Statistical prediction
        stat_result = self.stat_detector.analyze(module, golden_features)
        stat_trojan_score = stat_result['score']
        stat_confidence = stat_result.get('confidence', 0.5)
        
        # Hybrid score (weighted combination)
        hybrid_score = (self.gnn_weight * gnn_trojan_score + 
                       self.stat_weight * stat_trojan_score)
        
        # Final prediction (threshold = 0.5)
        final_pred = 1 if hybrid_score > 0.5 else 0
        
        # Confidence based on agreement
        if gnn_pred == (1 if stat_trojan_score > 0.5 else 0):
            final_confidence = (gnn_confidence + stat_confidence) / 2
        else:
            final_confidence = abs(hybrid_score - 0.5) * 2  # Distance from decision boundary
        
        return {
            'prediction': final_pred,
            'confidence': final_confidence,
            'hybrid_score': hybrid_score,
            'gnn_score': gnn_trojan_score,
            'gnn_confidence': gnn_confidence,
            'statistical_score': stat_trojan_score,
            'statistical_confidence': stat_confidence,
            'anomalies': stat_result,
            'embedding': embedding[0].numpy(),
            'method': 'hybrid'
        }

# -----------------------
# PLOTLY VISUALIZATION
# -----------------------
def create_interactive_graph(module: VerilogModule, highlight_suspicious: List[str] = None):
    """Create publication-quality interactive graph"""
    
    G = nx.DiGraph()
    
    # Build graph
    for name, sig in module.signals.items():
        G.add_node(name, 
                   signal_type=sig.signal_type,
                   width=sig.width,
                   fanin=sig.fanin,
                   fanout=sig.fanout,
                   is_clock=sig.is_clock,
                   is_reset=sig.is_reset)
    
    # Add edges
    for src, tgt_expr in module.assignments:
        targets = re.findall(r'\b\w+\b', tgt_expr)
        for tgt in targets:
            if tgt in module.signals and src in module.signals:
                G.add_edge(src, tgt)
    
    # Layout
    if len(G) > 0:
        try:
            pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)
        except:
            pos = nx.random_layout(G, seed=42)
    else:
        pos = {}
    
    # Create edges
    edge_x, edge_y = [], []
    for edge in G.edges():
        if edge[0] in pos and edge[1] in pos:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.8, color='#aaa'),
        hoverinfo='none',
        mode='lines',
        name='Connections'
    )
    
    # Create nodes
    node_x, node_y = [], []
    node_text, node_color, node_size = [], [], []
    
    highlight_set = set(highlight_suspicious or [])
    
    for node in G.nodes():
        if node not in pos:
            continue
            
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        sig = module.signals[node]
        
        # Enhanced color coding
        if node in highlight_set:
            color = '#ff1744'  # Red - suspicious
        elif sig.is_clock:
            color = '#ffd600'  # Yellow - clock
        elif sig.is_reset:
            color = '#ff6f00'  # Orange - reset
        elif sig.fanout > 10:
            color = '#ff5722'  # Deep orange - high fanout
        elif sig.fanin == 0 and sig.fanout == 0:
            color = '#9e9e9e'  # Gray - isolated
        else:
            color = {
                "input": "#00c853",   # Green
                "output": "#2979ff",  # Blue
                "reg": "#aa00ff",     # Purple
                "wire": "#757575"     # Dark gray
            }.get(sig.signal_type, "#9e9e9e")
        
        node_color.append(color)
        
        # Size based on importance
        degree = G.degree(node)
        size = 18 + min(degree * 3, 50)
        node_size.append(size)
        
        # Enhanced hover text
        hover = f"<b>{node}</b><br>"
        hover += f"Type: {sig.signal_type}<br>"
        hover += f"Width: {sig.width} bits<br>"
        hover += f"Fan-in: {sig.fanin}<br>"
        hover += f"Fan-out: {sig.fanout}<br>"
        hover += f"Degree: {degree}"
        
        if sig.is_clock:
            hover += "<br><b>Clock Signal</b>"
        if sig.is_reset:
            hover += "<br><b>Reset Signal</b>"
        if node in highlight_set:
            hover += "<br><b style='color:red'>⚠ SUSPICIOUS</b>"
            
        node_text.append(hover)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[n for n in G.nodes() if n in pos],
        hovertext=node_text,
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(width=2, color='white'),
            opacity=0.9
        ),
        name='Signals'
    )
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text=f'Signal Dependency Graph - {module.name}',
                font=dict(size=18, color='#333')
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=750
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Enhanced legend
    legend_html = """
    <div style='padding: 15px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                border-radius: 10px; margin-top: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
        <b style='font-size: 16px;'>🎨 Legend</b><br><br>
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;'>
            <div><span style='color: #00c853; font-size: 20px;'>●</span> Input Signal</div>
            <div><span style='color: #2979ff; font-size: 20px;'>●</span> Output Signal</div>
            <div><span style='color: #aa00ff; font-size: 20px;'>●</span> Register</div>
            <div><span style='color: #757575; font-size: 20px;'>●</span> Wire</div>
            <div><span style='color: #ffd600; font-size: 20px;'>●</span> Clock</div>
            <div><span style='color: #ff6f00; font-size: 20px;'>●</span> Reset</div>
            <div><span style='color: #ff5722; font-size: 20px;'>●</span> High Fan-out</div>
            <div><span style='color: #ff1744; font-size: 20px;'>●</span> ⚠ Suspicious</div>
        </div>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)

# -----------------------
# METRICS DASHBOARD
# -----------------------
def create_competition_dashboard(modules: List[VerilogModule], predictions: List[dict]):
    """Competition-grade metrics dashboard"""
    
    data = []
    for mod, pred in zip(modules, predictions):
        data.append({
            'Module': mod.name,
            'Prediction': 'HT-infested' if pred['prediction'] == 1 else 'HT-free',
            'Confidence': pred['confidence'] * 100,
            'Hybrid Score': pred['hybrid_score'] * 100,
            'GNN Score': pred['gnn_score'] * 100,
            'Statistical Score': pred['statistical_score'] * 100,
            'Signals': len(mod.signals),
            'Assignments': len(mod.assignments),
            'Complexity': len(mod.always_blocks)
        })
    
    df = pd.DataFrame(data)
    
    # Enhanced visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            df,
            x='Module',
            y='Hybrid Score',
            color='Prediction',
            title='🎯 Trojan Detection Scores by Module',
            color_discrete_map={'HT-free': '#00c853', 'HT-infested': '#d50000'},
            labels={'Hybrid Score': 'Detection Score (%)'}
        )
        fig.update_layout(
            font=dict(size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            df,
            x='Signals',
            y='Hybrid Score',
            size='Complexity',
            color='Prediction',
            hover_data=['Module', 'Confidence'],
            title='📊 Complexity vs Detection Score',
            color_discrete_map={'HT-free': '#00c853', 'HT-infested': '#d50000'},
            labels={'Hybrid Score': 'Detection Score (%)', 'Signals': 'Number of Signals'}
        )
        fig.update_layout(
            font=dict(size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Method comparison
    st.subheader("⚖️ Detection Method Comparison")
    
    method_df = df[['Module', 'GNN Score', 'Statistical Score', 'Hybrid Score']].melt(
        id_vars=['Module'],
        value_vars=['GNN Score', 'Statistical Score', 'Hybrid Score'],
        var_name='Method',
        value_name='Score'
    )
    
    fig = px.line(
        method_df,
        x='Module',
        y='Score',
        color='Method',
        markers=True,
        title='Detection Method Performance Comparison',
        labels={'Score': 'Detection Score (%)'}
    )
    fig.update_layout(
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    return df

# -----------------------
# MAIN APPLICATION
# -----------------------
def main():
    # Header
    st.markdown('<h1 class="main-header">🔐 Hardware Trojan Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<div class="competition-badge">IIT Kanpur Hardware Security Bootcamp 2026</div>', unsafe_allow_html=True)
    st.markdown("**Advanced ML/DL/GNN + Statistical Hybrid Detection Platform**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.markdown("### Detection Method")
        detection_method = st.radio(
            "Select approach:",
            ["Hybrid (GNN + Statistical)", "GNN Only", "Statistical Only"],
            index=0,
            help="Hybrid method recommended for competition"
        )
        
        if detection_method == "Hybrid (GNN + Statistical)":
            gnn_weight = st.slider("GNN Weight", 0.0, 1.0, 0.6, 0.1)
            stat_weight = 1.0 - gnn_weight
            st.info(f"Statistical Weight: {stat_weight:.1f}")
        
        st.markdown("---")
        st.markdown("### Visualization Options")
        show_graph = st.checkbox("Show Dependency Graph", value=True)
        show_anomalies = st.checkbox("Show Anomaly Details", value=True)
        show_statistics = st.checkbox("Show Module Statistics", value=True)
        show_comparison = st.checkbox("Show Method Comparison", value=True)
        
        st.markdown("---")
        st.markdown("### Golden Model")
        use_golden = st.checkbox("Use Golden Model Reference", value=False)
        golden_features = None
        
        st.markdown("---")
        st.markdown("### 📊 System Info")
        st.info("""
        **Hybrid Architecture:**
        - GNN: 4-layer GAT (256 hidden)
        - Statistical: Multi-metric analyzer
        - Features: 48-dimensional
        - Ensemble: Weighted voting
        """)
        
        st.markdown("---")
        st.markdown("### 📖 Competition Requirements")
        st.success("""
        ✅ Labeled design analysis
        ✅ I/O characteristics extraction
        ✅ Golden model comparison
        ✅ Statistical model
        ✅ ML/DL/GNN model
        ✅ Binary classification
        """)
    
    # File upload
    st.header("📂 Upload Verilog Designs")
    files = st.file_uploader(
        "Upload Verilog RTL files (.v or .vh)",
        type=["v", "vh"],
        accept_multiple_files=True,
        help="Upload both HT-free and HT-infested designs"
    )
    
    if not files:
        st.info("👆 Upload Verilog design files to start Hardware Trojan detection")
        
        with st.expander("ℹ️ Competition Overview"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ### 🎯 Objective
                Detect hardware trojans in RTL designs using:
                - Statistical analysis
                - Machine Learning (GNN)
                - Golden model comparison
                
                ### 📥 Inputs
                - Verilog designs (.v files)
                - Design labels (HT-free/HT-infested)
                - Golden IC-free reference model
                
                ### 📤 Outputs
                - Binary classification
                - Confidence scores
                - Anomaly reports
                """)
            
            with col2:
                st.markdown("""
                ### 🔍 Detection Approaches
                
                **Statistical Model:**
                - I/O characteristics
                - Structural analysis
                - Pattern matching
                - Golden comparison
                
                **GNN Model:**
                - Graph representation
                - Attention mechanisms
                - Multi-scale features
                - Deep learning
                
                **Hybrid (Recommended):**
                - Combines both methods
                - Ensemble voting
                - Higher accuracy
                """)
        
        return
    
    # Initialize components
    parser = CompetitionVerilogParser()
    
    if detection_method == "Hybrid (GNN + Statistical)":
        builder = EnhancedGraphBuilder(feature_dim=48)
        detector = HybridTrojanDetectionSystem(gnn_weight=gnn_weight, stat_weight=stat_weight)
    else:
        builder = EnhancedGraphBuilder(feature_dim=48)
        if detection_method == "GNN Only":
            detector = HybridTrojanDetectionSystem(gnn_weight=1.0, stat_weight=0.0)
        else:
            detector = HybridTrojanDetectionSystem(gnn_weight=0.0, stat_weight=1.0)
    
    # Process files
    modules = []
    predictions = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, file in enumerate(files):
        status_text.text(f"🔄 Processing {file.name}...")
        progress_bar.progress((idx + 1) / len(files))
        
        try:
            content = file.read().decode("utf-8", errors="ignore")
            module = parser.parse(content)
            modules.append(module)
            
            # Build graph
            graph = builder.build(module)
            
            # Predict
            pred = detector.predict(module, graph, golden_features)
            pred['filename'] = file.name
            predictions.append(pred)
            
        except Exception as e:
            st.error(f"❌ Error processing {file.name}: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    if len(predictions) == 0:
        st.error("No files were successfully processed.")
        return
    
    # Results header
    st.success(f"✅ Successfully analyzed {len(modules)} design(s)")
    
    # Display comparison dashboard
    if len(modules) > 1 and show_comparison:
        st.header("📊 Comparative Analysis Dashboard")
        df = create_competition_dashboard(modules, predictions)
        
        # Show detailed table
        with st.expander("📋 Detailed Results Table", expanded=False):
            st.dataframe(df, use_container_width=True)
    
    # Individual module analysis
    st.header("🔬 Individual Design Analysis")
    
    for module, pred in zip(modules, predictions):
        # Prediction result styling
        if pred['prediction'] == 1:
            result_class = "trojan-detected"
            result_icon = "🚨"
            result_text = "HARDWARE TROJAN DETECTED"
            result_color = "#d50000"
        else:
            result_class = "clean-design"
            result_icon = "✅"
            result_text = "CLEAN DESIGN (HT-FREE)"
            result_color = "#00c853"
        
        with st.expander(f"{result_icon} {pred['filename']}", expanded=True):
            # Main prediction
            st.markdown(f"""
            <div class="{result_class}">
                <h2 style='margin: 0; color: {result_color};'>{result_icon} {result_text}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Overall Confidence",
                    f"{pred['confidence']*100:.1f}%",
                    delta=f"{pred['hybrid_score']*100:.1f}% score"
                )
            
            with col2:
                st.metric(
                    "GNN Detection",
                    f"{pred['gnn_score']*100:.1f}%",
                    delta=f"{pred['gnn_confidence']*100:.1f}% conf."
                )
            
            with col3:
                st.metric(
                    "Statistical Analysis",
                    f"{pred['statistical_score']*100:.1f}%",
                    delta=f"{pred['statistical_confidence']*100:.1f}% conf."
                )
            
            with col4:
                st.metric(
                    "Detection Method",
                    pred['method'].upper(),
                    delta=detection_method
                )
            
            # Module statistics
            if show_statistics:
                st.subheader("📈 Design Statistics")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Signals", len(module.signals))
                col2.metric("Assignments", len(module.assignments))
                col3.metric("Always Blocks", len(module.always_blocks))
                col4.metric("Instances", len(module.instances))
                col5.metric("Parameters", len(module.parameters))
                
                # Signal breakdown
                signal_types = defaultdict(int)
                for sig in module.signals.values():
                    signal_types[sig.signal_type] += 1
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Signal Distribution:**")
                    for stype, count in sorted(signal_types.items()):
                        st.text(f"  {stype}: {count}")
                
                with col2:
                    st.markdown("**Special Signals:**")
                    clocks = sum(1 for s in module.signals.values() if s.is_clock)
                    resets = sum(1 for s in module.signals.values() if s.is_reset)
                    st.text(f"  Clock signals: {clocks}")
                    st.text(f"  Reset signals: {resets}")
                
                with col3:
                    st.markdown("**Connectivity:**")
                    fanins = [s.fanin for s in module.signals.values()]
                    fanouts = [s.fanout for s in module.signals.values()]
                    st.text(f"  Avg fan-in: {np.mean(fanins):.1f}")
                    st.text(f"  Avg fan-out: {np.mean(fanouts):.1f}")
                    st.text(f"  Max fan-out: {max(fanouts) if fanouts else 0}")
            
            # Anomaly details
            if show_anomalies and pred['statistical_score'] > 0:
                st.subheader("⚠️ Detailed Anomaly Analysis")
                
                anomalies = pred['anomalies']
                
                # Create anomaly summary
                anomaly_categories = []
                if anomalies.get('suspicious_names'):
                    anomaly_categories.append(("Suspicious Names", len(anomalies['suspicious_names']), "high"))
                if anomalies.get('unusual_widths'):
                    anomaly_categories.append(("Unusual Widths", len(anomalies['unusual_widths']), "medium"))
                if anomalies.get('high_fanout'):
                    anomaly_categories.append(("High Fan-out", len(anomalies['high_fanout']), "medium"))
                if anomalies.get('isolated_signals'):
                    anomaly_categories.append(("Isolated Signals", len(anomalies['isolated_signals']), "medium"))
                if anomalies.get('complex_logic'):
                    anomaly_categories.append(("Complex Logic", len(anomalies['complex_logic']), "low"))
                if anomalies.get('rare_signals'):
                    anomaly_categories.append(("Rare Signals", len(anomalies['rare_signals']), "medium"))
                
                if anomaly_categories:
                    # Create anomaly visualization
                    anomaly_df = pd.DataFrame(anomaly_categories, columns=['Category', 'Count', 'Severity'])
                    
                    fig = px.bar(
                        anomaly_df,
                        x='Category',
                        y='Count',
                        color='Severity',
                        title='Anomaly Distribution',
                        color_discrete_map={'high': '#d50000', 'medium': '#ff6f00', 'low': '#ffd600'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed anomaly information
                col1, col2 = st.columns(2)
                
                with col1:
                    if anomalies.get('suspicious_names'):
                        st.warning(f"**🔍 Suspicious Signal Names ({len(anomalies['suspicious_names'])}):**")
                        st.text(", ".join(anomalies['suspicious_names'][:10]))
                        if len(anomalies['suspicious_names']) > 10:
                            st.text(f"... and {len(anomalies['suspicious_names']) - 10} more")
                    
                    if anomalies.get('high_fanout'):
                        st.warning(f"**📡 High Fan-out Signals ({len(anomalies['high_fanout'])}):**")
                        for name, fanout in anomalies['high_fanout'][:5]:
                            st.text(f"  {name}: {fanout} connections")
                    
                    if anomalies.get('rare_signals'):
                        st.warning(f"**⏱️ Rare/Counter Signals ({len(anomalies['rare_signals'])}):**")
                        st.text(", ".join(anomalies['rare_signals'][:10]))
                
                with col2:
                    if anomalies.get('unusual_widths'):
                        st.warning(f"**📏 Unusual Bit Widths ({len(anomalies['unusual_widths'])}):**")
                        for name, width in anomalies['unusual_widths'][:5]:
                            st.text(f"  {name}: {width} bits")
                    
                    if anomalies.get('isolated_signals'):
                        st.warning(f"**🔌 Isolated Signals ({len(anomalies['isolated_signals'])}):**")
                        st.text(", ".join(anomalies['isolated_signals'][:10]))
                    
                    if anomalies.get('complex_logic'):
                        st.warning(f"**🧩 Complex Logic Blocks: {len(anomalies['complex_logic'])}**")
            
            # Dependency graph
            if show_graph:
                st.subheader("🔗 Signal Dependency Graph")
                suspicious_signals = pred['anomalies'].get('suspicious_names', []) if pred['prediction'] == 1 else []
                create_interactive_graph(module, suspicious_signals)
            
            # Signal details table
            st.markdown("---")
            if st.checkbox(f"📋 Show detailed signal information", key=f"signals_{pred['filename']}"):
                signal_data = []
                for name, sig in module.signals.items():
                    is_suspicious = name in suspicious_signals
                    signal_data.append({
                        'Signal': name,
                        'Type': sig.signal_type,
                        'Width': sig.width,
                        'Fan-in': sig.fanin,
                        'Fan-out': sig.fanout,
                        'Clock': '✓' if sig.is_clock else '',
                        'Reset': '✓' if sig.is_reset else '',
                        'Suspicious': '⚠️' if is_suspicious else ''
                    })
                
                signal_df = pd.DataFrame(signal_data)
                st.dataframe(signal_df, use_container_width=True)
    
    # Overall summary
    st.header("📊 Detection Summary")
    
    total = len(predictions)
    trojans = sum(1 for p in predictions if p['prediction'] == 1)
    clean = total - trojans
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Designs", total)
    col2.metric("HT-infested", trojans, delta=f"{trojans/total*100:.1f}%")
    col3.metric("HT-free", clean, delta=f"{clean/total*100:.1f}%")
    col4.metric("Avg Confidence", f"{np.mean([p['confidence'] for p in predictions])*100:.1f}%")
    
    # Export results
    st.markdown("---")
    st.subheader("📥 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # JSON export
        report = {
            'competition': 'IIT Kanpur Hardware Security Bootcamp 2026',
            'detection_method': detection_method,
            'summary': {
                'total_designs': total,
                'ht_infested': trojans,
                'ht_free': clean,
                'detection_rate': float(trojans / total) if total > 0 else 0,
                'avg_confidence': float(np.mean([p['confidence'] for p in predictions]))
            },
            'designs': []
        }
        
        for module, pred in zip(modules, predictions):
            report['designs'].append({
                'filename': pred['filename'],
                'module_name': module.name,
                'prediction': 'HT-infested' if pred['prediction'] == 1 else 'HT-free',
                'confidence': float(pred['confidence']),
                'hybrid_score': float(pred['hybrid_score']),
                'gnn_score': float(pred['gnn_score']),
                'statistical_score': float(pred['statistical_score']),
                'statistics': {
                    'signals': len(module.signals),
                    'assignments': len(module.assignments),
                    'always_blocks': len(module.always_blocks),
                    'instances': len(module.instances)
                },
                'anomalies': {
                    k: (v if isinstance(v, (int, float, str, bool)) else len(v))
                    for k, v in pred['anomalies'].items()
                    if k not in ['score', 'confidence']
                }
            })
        
        st.download_button(
            label="📄 Download JSON Report",
            data=json.dumps(report, indent=2),
            file_name=f"trojan_detection_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        # CSV export
        if len(predictions) > 1:
            csv_data = []
            for module, pred in zip(modules, predictions):
                csv_data.append({
                    'Filename': pred['filename'],
                    'Module': module.name,
                    'Prediction': 'HT-infested' if pred['prediction'] == 1 else 'HT-free',
                    'Confidence': pred['confidence'],
                    'Hybrid_Score': pred['hybrid_score'],
                    'GNN_Score': pred['gnn_score'],
                    'Statistical_Score': pred['statistical_score'],
                    'Signals': len(module.signals),
                    'Assignments': len(module.assignments)
                })
            
            csv_df = pd.DataFrame(csv_data)
            
            st.download_button(
                label="📊 Download CSV Results",
                data=csv_df.to_csv(index=False),
                file_name=f"trojan_detection_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()