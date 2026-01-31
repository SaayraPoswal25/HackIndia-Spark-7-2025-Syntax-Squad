import React, { useEffect, useState } from 'react';
import { Activity, Shield, Users, Server, Database, Lock, RefreshCw, Upload, Download } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import axios from 'axios';
import { motion } from 'framer-motion';

const API_URL = 'http://localhost:8000';

interface Status {
    round: number;
    pending_updates: number;
    required_updates: number;
    clients_online: number;
}

const Dashboard = () => {
    const [status, setStatus] = useState<Status | null>(null);
    const [loading, setLoading] = useState(true);
    const [logs, setLogs] = useState<string[]>([]);

    // Mock data for the chart since we don't have historical data persistence in this demo
    const [data, setData] = useState([
        { name: 'Round 1', accuracy: 65 },
        { name: 'Round 2', accuracy: 72 },
        { name: 'Round 3', accuracy: 78 },
        { name: 'Round 4', accuracy: 82 },
        { name: 'Round 5', accuracy: 85 },
    ]);

    const fetchStatus = async () => {
        try {
            const res = await axios.get(`${API_URL}/status`);
            setStatus(res.data);
            setLoading(false);
        } catch (error) {
            console.error("Failed to fetch status", error);
        }
    };

    useEffect(() => {
        fetchStatus();
        const interval = setInterval(fetchStatus, 5000); // Poll every 5 seconds

        // Add some initial logs
        setLogs(prev => [
            ...prev,
            `[${new Date().toLocaleTimeString()}] System initialized.`,
            `[${new Date().toLocaleTimeString()}] Connected to Secure Aggregation Server via HTTPS (Simulated).`,
            `[${new Date().toLocaleTimeString()}] Waiting for client updates...`
        ]);

        return () => clearInterval(interval);
    }, []);

    return (
        <div className="min-h-screen bg-background text-foreground p-8 font-sans">
            <header className="mb-12 flex justify-between items-center">
                <div>
                    <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-600 text-transparent bg-clip-text">
                        Hospintel
                    </h1>
                    <p className="text-muted-foreground mt-2">Hospital Intelligence • Secure Federated Learning</p>
                </div>
                <div className="flex gap-4">
                    <div className="flex items-center gap-2 px-4 py-2 bg-card rounded-full border border-border">
                        <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                        <span className="text-sm font-medium">System Online</span>
                    </div>
                </div>
            </header>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                <StatCard
                    icon={<Activity className="text-blue-400" />}
                    title="Current Round"
                    value={status?.round || 1}
                    sub="Training Phase"
                />
                <StatCard
                    icon={<Users className="text-purple-400" />}
                    title="Pending Updates"
                    value={`${status?.pending_updates || 0} / ${status?.required_updates || 2}`}
                    sub="Waiting for aggregation"
                />
                <StatCard
                    icon={<Shield className="text-green-400" />}
                    title="Security Level"
                    value="High"
                    sub="AES-256 + RSA Encryption"
                />
                <StatCard
                    icon={<Database className="text-pink-400" />}
                    title="Global Model"
                    value="VGG16"
                    sub="Accuracy: 85% (Est.)"
                />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <div className="lg:col-span-2 bg-card border border-border rounded-xl p-6 shadow-sm">
                    <h3 className="text-xl font-semibold mb-6 flex items-center gap-2">
                        <Activity className="w-5 h-5 text-blue-400" />
                        Model Performance
                    </h3>
                    <div className="h-[300px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={data}>
                                <defs>
                                    <linearGradient id="colorAccuracy" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#8884d8" stopOpacity={0.8} />
                                        <stop offset="95%" stopColor="#8884d8" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                                <XAxis dataKey="name" stroke="#666" />
                                <YAxis stroke="#666" />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155' }}
                                    itemStyle={{ color: '#e2e8f0' }}
                                />
                                <Area type="monotone" dataKey="accuracy" stroke="#8884d8" fillOpacity={1} fill="url(#colorAccuracy)" />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="bg-card border border-border rounded-xl p-6 shadow-sm flex flex-col h-full">
                    <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                        <Lock className="w-5 h-5 text-green-400" />
                        Security Audit Log
                    </h3>
                    <div className="flex-1 overflow-y-auto space-y-3 pr-2 max-h-[300px] font-mono text-sm">
                        {logs.map((log, i) => (
                            <motion.div
                                key={i}
                                initial={{ opacity: 0, x: -10 }}
                                animate={{ opacity: 1, x: 0 }}
                                className="text-muted-foreground border-l-2 border-slate-700 pl-3 py-1"
                            >
                                {log}
                            </motion.div>
                        ))}
                    </div>
                </div>
            </div>

            <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
                <FeatureCard
                    icon={<Upload className="w-6 h-6" />}
                    title="Encrypted Uploads"
                    desc="Client gradients are encrypted with Hybrid RSA+AES before transmission."
                />
                <FeatureCard
                    icon={<Server className="w-6 h-6" />}
                    title="Secure Aggregation"
                    desc="Server averages weights in memory without storing individual updates persistently."
                />
                <FeatureCard
                    icon={<Shield className="w-6 h-6" />}
                    title="PII Protection"
                    desc="Raw data never leaves the client device. Only abstract weight updates are shared."
                />
            </div>
        </div>
    );
};

const StatCard = ({ icon, title, value, sub }: any) => (
    <motion.div
        whileHover={{ y: -5 }}
        className="bg-card border border-border rounded-xl p-6 shadow-lg hover:shadow-xl transition-all"
    >
        <div className="flex items-start justify-between">
            <div>
                <p className="text-sm font-medium text-muted-foreground">{title}</p>
                <h3 className="text-2xl font-bold mt-2">{value}</h3>
                <p className="text-xs text-muted-foreground mt-1">{sub}</p>
            </div>
            <div className="p-3 bg-secondary rounded-lg">
                {icon}
            </div>
        </div>
    </motion.div>
);

const FeatureCard = ({ icon, title, desc }: any) => (
    <div className="bg-card/50 border border-border rounded-xl p-6">
        <div className="flex items-center gap-3 mb-3 text-primary">
            {icon}
            <h4 className="font-semibold">{title}</h4>
        </div>
        <p className="text-sm text-muted-foreground leading-relaxed">
            {desc}
        </p>
    </div>
);

export default Dashboard;
