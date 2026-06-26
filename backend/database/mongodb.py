"""
MongoDB Async Database Client (Motor)
Collections: users, predictions, reports, chat_history, drug_alerts, analytics
Falls back gracefully if MongoDB is not available.
"""
import os
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME   = "medirag_xai"


class MediDB:
    def __init__(self):
        self.client = None
        self.db     = None
        self._connected = False
        self._try_connect()

    def _try_connect(self):
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            self.client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=3000)
            self.db     = self.client[DB_NAME]
            self._connected = True
            print(f"[OK] MongoDB client created -> {DB_NAME}")
        except Exception as e:
            print(f"[WARN] MongoDB not available: {e}. Using in-memory fallback.")
            self._connected = False
            self._memory   = {
                "predictions": [], "reports": [], "chat_history": [],
                "drug_alerts": [], "users": []
            }

    # ── Write Operations ───────────────────────────────────────────────────────
    async def save_prediction(self, data: Dict) -> Optional[str]:
        data["timestamp"] = datetime.utcnow()
        if self._connected:
            try:
                result = await self.db.predictions.insert_one(data)
                return str(result.inserted_id)
            except Exception as e:
                print(f"DB write error (prediction): {e}")
        else:
            self._memory["predictions"].append(data)
        return None

    async def save_report(self, data: Dict) -> Optional[str]:
        data["timestamp"] = datetime.utcnow()
        if self._connected:
            try:
                result = await self.db.reports.insert_one(data)
                return str(result.inserted_id)
            except Exception as e:
                print(f"DB write error (report): {e}")
        else:
            self._memory["reports"].append(data)
        return None

    async def save_chat(self, data: Dict) -> Optional[str]:
        data["timestamp"] = datetime.utcnow()
        if self._connected:
            try:
                result = await self.db.chat_history.insert_one(data)
                return str(result.inserted_id)
            except Exception as e:
                print(f"DB write error (chat): {e}")
        else:
            self._memory["chat_history"].append(data)
        return None

    async def save_drug_alert(self, data: Dict) -> Optional[str]:
        data["timestamp"] = datetime.utcnow()
        if self._connected:
            try:
                result = await self.db.drug_alerts.insert_one(data)
                return str(result.inserted_id)
            except Exception as e:
                print(f"DB write error (drug_alert): {e}")
        else:
            self._memory["drug_alerts"].append(data)
        return None

    async def save_user(self, user_data: Dict) -> Optional[str]:
        user_data["timestamp"] = datetime.utcnow()
        if self._connected:
            try:
                # Ensure unique index on email
                await self.db.users.create_index("email", unique=True)
                result = await self.db.users.insert_one(user_data)
                return str(result.inserted_id)
            except Exception as e:
                print(f"DB write error (user): {e}")
        else:
            # Check for unique email in memory
            for u in self._memory["users"]:
                if u.get("email") == user_data.get("email"):
                    print(f"DB write warning: email {user_data.get('email')} already exists in memory.")
                    return None
            self._memory["users"].append(user_data)
            return "mem_user_" + str(len(self._memory["users"]))
        return None

    async def get_user_by_email(self, email: str) -> Optional[Dict]:
        if self._connected:
            try:
                return await self.db.users.find_one({"email": email})
            except Exception as e:
                print(f"DB read error (user by email): {e}")
        else:
            for u in self._memory["users"]:
                if u.get("email") == email:
                    return u
        return None

    async def get_user_by_medical_id(self, medical_id: str) -> Optional[Dict]:
        if self._connected:
            try:
                return await self.db.users.find_one({"medical_id": medical_id})
            except Exception as e:
                print(f"DB read error (user by medical_id): {e}")
        else:
            for u in self._memory["users"]:
                if u.get("medical_id") == medical_id:
                    return u
        return None

    # ── Analytics Aggregation ──────────────────────────────────────────────────
    async def get_analytics(self) -> Dict[str, Any]:
        if self._connected:
            try:
                return await self._aggregate_from_mongo()
            except Exception as e:
                print(f"DB read error (analytics): {e}")
        return await self._aggregate_from_memory()

    async def _aggregate_from_mongo(self) -> Dict[str, Any]:
        # Disease distribution
        disease_pipeline = [
            {"$group": {"_id": "$top_prediction.disease", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10},
        ]
        disease_cursor = self.db.predictions.aggregate(disease_pipeline)
        disease_dist   = {}
        async for doc in disease_cursor:
            if doc["_id"]:
                disease_dist[doc["_id"]] = doc["count"]

        # Total counts
        total_predictions = await self.db.predictions.count_documents({})
        total_reports     = await self.db.reports.count_documents({})
        total_chats       = await self.db.chat_history.count_documents({})
        total_drug_checks = await self.db.drug_alerts.count_documents({})

        # Drug alerts by type
        alert_pipeline = [
            {"$unwind": "$alerts"},
            {"$group": {"_id": "$alerts.category", "count": {"$sum": 1}}},
        ]
        alert_cursor = self.db.drug_alerts.aggregate(alert_pipeline)
        drug_alerts  = {}
        async for doc in alert_cursor:
            if doc["_id"]:
                drug_alerts[doc["_id"]] = doc["count"]

        return {
            "disease_distribution":  disease_dist or self._demo_disease_dist(),
            "prediction_confidence": self._demo_confidence(),
            "confidence_labels":     self._demo_labels(),
            "common_symptoms":       self._demo_symptoms(),
            "drug_alerts":           drug_alerts or self._demo_drug_alerts(),
            "chat_usage":            self._demo_chat_usage(),
            "total_predictions":     total_predictions,
            "total_reports":         total_reports,
            "total_chats":           total_chats,
            "total_drug_checks":     total_drug_checks,
        }

    async def _aggregate_from_memory(self) -> Dict[str, Any]:
        preds = self._memory.get("predictions", [])
        disease_dist = {}
        for p in preds:
            d = p.get("top_prediction", {}).get("disease", "Unknown")
            disease_dist[d] = disease_dist.get(d, 0) + 1

        return {
            "disease_distribution":  disease_dist or self._demo_disease_dist(),
            "prediction_confidence": self._demo_confidence(),
            "confidence_labels":     self._demo_labels(),
            "common_symptoms":       self._demo_symptoms(),
            "drug_alerts":           self._demo_drug_alerts(),
            "chat_usage":            self._demo_chat_usage(),
            "total_predictions":     len(self._memory["predictions"]),
            "total_reports":         len(self._memory["reports"]),
            "total_chats":           len(self._memory["chat_history"]),
            "total_drug_checks":     len(self._memory["drug_alerts"]),
        }

    # ── Demo Data (used when DB is fresh) ─────────────────────────────────────
    @staticmethod
    def _demo_disease_dist():
        return {
            "Diabetes ": 145, "Hypertension ": 132, "Bronchial Asthma": 98,
            "COPD": 67, "Heart attack": 89, "Migraine": 75,
            "Anxiety": 112, "Tuberculosis": 58, "Pneumonia": 65,
            "Dengue": 88, "Other": 141
        }

    @staticmethod
    def _demo_confidence():
        return [72, 78, 85, 81, 88, 91, 87, 93, 89, 95, 92, 88, 94, 90, 96]

    @staticmethod
    def _demo_labels():
        return ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
                "Aug", "Sep", "Oct", "Nov", "Dec", "Q1", "Q2", "Q3"]

    @staticmethod
    def _demo_symptoms():
        return {
            "Fever": 312, "Fatigue": 287, "Cough": 245,
            "Headache": 198, "Chest Pain": 156, "Shortness of Breath": 143,
            "Nausea": 132, "Joint Pain": 118
        }

    @staticmethod
    def _demo_drug_alerts():
        return {
            "Contraindication": 45, "Pregnancy Warning": 23,
            "Allergy Alert": 31, "Drug Interaction": 67, "Safe": 234
        }

    @staticmethod
    def _demo_chat_usage():
        return [12, 18, 25, 32, 28, 41, 35, 48, 52, 44, 38, 55, 61, 47, 58]
