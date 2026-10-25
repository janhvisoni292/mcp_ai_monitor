
import json
import os
import sys

# Ensure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_tools import tools

class SelfAuditingAgent:
    def __init__(self, sensitivity="Medium"):
        self.name = "MCP-Audit-Core-v1"
        self.sensitivity_map = {
            "Low": 0.5,
            "Medium": 1.0, 
            "High": 1.5
        }
        self.multiplier = self.sensitivity_map.get(sensitivity, 1.0)
        print(f"[{self.name}] Initialized with {sensitivity} sensitivity (Multiplier: {self.multiplier})")

    def run_audit(self):
        """
        Main execution loop for the agent.
        """
        report = {
            "status": "PASS", 
            "issues": [],
            "metrics": {
                "fail_rate": 0, # Legacy key for UI
                "pos_rate": 0
            },
            "risk_score": 0,
            "config_used": {"sensitivity": self.multiplier}
        }

        print(f"[{self.name}] Starting Audit Cycle...")

        # --- STEP 1: Check General Health ---
        health = tools.get_current_model_health()
        report["metrics"]["health"] = health
        
        # Rule: If confidence drops below 0.6, flag it.
        if health.get("avg_confidence", 1.0) < 0.6:
            report["issues"].append("Low Average Confidence detected.")
            report["risk_score"] += (20 * self.multiplier)

        # --- STEP 2: Check Drift ---
        drift = tools.check_feature_drift(recent_window=50)
        report["metrics"]["drift"] = drift
        
        drift_count = 0
        for feature, data in drift.items():
            if data["is_drifting"]:
                drift_count += 1
                report["issues"].append(f"Major Data Drift detected in {feature} (Score: {data['drift_score']:.2f})")
        
        if drift_count > 0:
            report["risk_score"] += (drift_count * 15 * self.multiplier) 

        # --- STEP 3: Smart Performance Check ---
        # Get recent performance from health data
        accuracy = health.get("estimated_accuracy")
        
        # Get recent prediction stats
        recents = tools.get_recent_predictions(limit=50)
        if recents:
            # Check for label imbalance (formerly "failure rate")
            # We only penalize this if accuracy is also low or unknown.
            positives = [r for r in recents if r['prediction'] == 1]
            pos_rate = len(positives) / len(recents)
            report["metrics"]["pos_rate"] = pos_rate
            report["metrics"]["fail_rate"] = pos_rate # Legacy alias for dashboard UI

            # If accuracy is known and HIGH, we reduce the total risk score (Incentive for good models)
            if accuracy is not None and accuracy > 0.9:
                report["risk_score"] *= 0.25 # Huge discount for high accuracy
                report["issues"] = [opt for opt in report["issues"] if "Drift" not in opt] # Accuracy trumps drift
                if not report["issues"]:
                    report["recommendation"] = "Model is performing excellently with high accuracy."

            # But if accuracy is LOW or Unknown, and labeling is highly biased (>80%)
            elif pos_rate > 0.8 or pos_rate < 0.05:
                report["issues"].append(f"Highly skewed predictions ({pos_rate:.1%}). Possible data bias.")
                report["risk_score"] += (15 * self.multiplier)

        # Round risk score
        report["risk_score"] = int(report["risk_score"])

        # --- Final Scoring ---
        if report["risk_score"] > 40:
            report["status"] = "CRITICAL"
            report["recommendation"] = "Immediate Repair Required. Run Auto-Heal."
        elif report["risk_score"] > 10:
            report["status"] = "WARNING"
            report["recommendation"] = "Monitor closely. Check data headers."
        
        print(f"[{self.name}] Audit Complete. Status: {report['status']}")
        return report

    def attempt_repair(self):
        """
        Calls the Repair Kit to fix the system.
        """
        from mcp_tools import repair_kit
        return repair_kit.perform_auto_repair()


if __name__ == "__main__":
    agent = SelfAuditingAgent()
    result = agent.run_audit()
    print("\n--- FINAL AGENT REPORT ---")
    print(json.dumps(result, indent=2))
