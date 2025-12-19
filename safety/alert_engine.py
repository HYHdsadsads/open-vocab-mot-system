"""工业安全预警规则引擎"""
from collections import defaultdict


class SafetyAlertEngine:
    def __init__(self):
        self.violation_records = defaultdict(lambda: {"count": 0, "duration": 0})
        self.alert_thresholds = {
            "no_helmet": {"count": 3, "duration": 300},  # 3次或持续300帧
            "out_of_region": {"count": 5, "duration": 100},
            "fast_movement": {"count": 2, "duration": 0}
        }

    def check_violations(self, tracks):
        alerts = []
        for track in tracks:
            # 检查未戴安全帽
            if track.class_name == "person" and "helmet" not in track.attributes:
                self.violation_records[track.id]["count"] += 1
                self.violation_records[track.id]["duration"] += 1

                if (self.violation_records[track.id]["count"] >= self.alert_thresholds["no_helmet"]["count"] or
                        self.violation_records[track.id]["duration"] >= self.alert_thresholds["no_helmet"]["duration"]):
                    alerts.append({
                        "type": "no_helmet",
                        "track_id": track.id,
                        "message": f"工人ID:{track.id}持续未佩戴安全帽"
                    })
        return alerts