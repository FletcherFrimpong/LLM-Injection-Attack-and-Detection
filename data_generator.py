#!/usr/bin/env python3
"""
Synthetic Data Generator for AI Training
Generates realistic but fake sensitive data for training purposes
"""

import json
import random
import string
from datetime import datetime, timedelta
from typing import Dict, List, Any
import hashlib
import uuid

class SyntheticDataGenerator:
    def __init__(self):
        self.customer_data = []
        self.financial_records = []
        self.company_secrets = []
        self.proprietary_algorithms = []
        self.security_protocols = []
        
    def generate_customer_personal_data(self, num_records: int = 100) -> List[Dict]:
        """Generate synthetic customer personal data"""
        first_names = [
            "John", "Jane", "Michael", "Sarah", "David", "Emily", "Robert", "Lisa",
            "James", "Jennifer", "William", "Jessica", "Richard", "Amanda", "Thomas",
            "Nicole", "Christopher", "Stephanie", "Daniel", "Melissa", "Matthew",
            "Ashley", "Anthony", "Elizabeth", "Mark", "Megan", "Donald", "Lauren",
            "Steven", "Rachel", "Paul", "Kimberly", "Andrew", "Heather", "Joshua",
            "Michelle", "Kenneth", "Tiffany", "Kevin", "Christine", "Brian", "Laura"
        ]
        
        last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
            "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
            "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
            "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
            "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
            "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green"
        ]
        
        companies = [
            "TechCorp", "GlobalSoft", "DataFlow", "CyberNet", "SecureSys", "CloudTech",
            "DigitalEdge", "NetSecure", "InfoGuard", "SafeNet", "TrustTech", "SecureFlow",
            "DataGuard", "CyberSafe", "NetProtect", "InfoSecure", "SafeData", "TrustNet"
        ]
        
        domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "company.com"]
        
        customers = []
        for i in range(num_records):
            first_name = random.choice(first_names)
            last_name = random.choice(last_names)
            email = f"{first_name.lower()}.{last_name.lower()}@{random.choice(domains)}"
            
            # Generate realistic phone number
            phone = f"+1-{random.randint(200, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
            
            # Generate realistic address
            street_number = random.randint(100, 9999)
            street_names = ["Main St", "Oak Ave", "Elm St", "Pine Rd", "Cedar Ln", "Maple Dr"]
            street = f"{street_number} {random.choice(street_names)}"
            city = random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia"])
            state = random.choice(["NY", "CA", "IL", "TX", "AZ", "PA"])
            zip_code = f"{random.randint(10000, 99999)}"
            
            customer = {
                "customer_id": f"CUST-{str(i+1).zfill(6)}",
                "first_name": first_name,
                "last_name": last_name,
                "email": email,
                "phone": phone,
                "address": {
                    "street": street,
                    "city": city,
                    "state": state,
                    "zip_code": zip_code,
                    "country": "USA"
                },
                "company": random.choice(companies),
                "registration_date": (datetime.now() - timedelta(days=random.randint(1, 1000))).strftime("%Y-%m-%d"),
                "status": random.choice(["active", "inactive", "suspended"]),
                "preferences": {
                    "newsletter": random.choice([True, False]),
                    "marketing_emails": random.choice([True, False]),
                    "two_factor_auth": random.choice([True, False])
                }
            }
            customers.append(customer)
        
        self.customer_data = customers
        return customers
    
    def generate_financial_records(self, num_records: int = 200) -> List[Dict]:
        """Generate synthetic financial records"""
        transaction_types = ["purchase", "refund", "transfer", "deposit", "withdrawal", "fee"]
        merchants = [
            "Amazon", "Walmart", "Target", "Starbucks", "McDonald's", "Netflix",
            "Spotify", "Uber", "Lyft", "DoorDash", "GrubHub", "Apple Store",
            "Google Play", "Microsoft", "Adobe", "Zoom", "Slack", "Dropbox"
        ]
        
        financial_records = []
        for i in range(num_records):
            transaction_type = random.choice(transaction_types)
            amount = round(random.uniform(1.00, 1000.00), 2)
            
            # Generate realistic transaction ID
            transaction_id = f"TXN-{datetime.now().strftime('%Y%m%d')}-{str(i+1).zfill(6)}"
            
            # Generate realistic account numbers
            account_number = f"{random.randint(100000000, 999999999)}"
            routing_number = f"{random.randint(100000000, 999999999)}"
            
            record = {
                "transaction_id": transaction_id,
                "account_number": account_number,
                "routing_number": routing_number,
                "transaction_type": transaction_type,
                "amount": amount,
                "currency": "USD",
                "merchant": random.choice(merchants) if transaction_type == "purchase" else None,
                "description": f"{transaction_type.title()} transaction",
                "timestamp": (datetime.now() - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d %H:%M:%S"),
                "status": random.choice(["completed", "pending", "failed"]),
                "balance_after": round(random.uniform(100.00, 50000.00), 2),
                "location": {
                    "city": random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]),
                    "state": random.choice(["NY", "CA", "IL", "TX", "AZ"]),
                    "country": "USA"
                }
            }
            financial_records.append(record)
        
        self.financial_records = financial_records
        return financial_records
    
    def generate_company_secrets(self, num_records: int = 50) -> List[Dict]:
        """Generate synthetic internal company secrets"""
        secret_types = [
            "product_roadmap", "acquisition_plan", "layoff_plan", "salary_data",
            "patent_filing", "trade_secret", "contract_terms", "board_meeting_minutes",
            "financial_forecast", "competitor_analysis", "security_breach", "legal_case"
        ]
        
        departments = [
            "Engineering", "Sales", "Marketing", "Finance", "HR", "Legal",
            "Operations", "Product", "Customer Support", "Research & Development"
        ]
        
        company_secrets = []
        for i in range(num_records):
            secret_type = random.choice(secret_types)
            department = random.choice(departments)
            
            # Generate realistic secret content
            secret_content = self._generate_secret_content(secret_type)
            
            secret = {
                "secret_id": f"SEC-{str(i+1).zfill(6)}",
                "type": secret_type,
                "department": department,
                "classification": random.choice(["confidential", "secret", "top_secret"]),
                "content": secret_content,
                "created_date": (datetime.now() - timedelta(days=random.randint(1, 1000))).strftime("%Y-%m-%d"),
                "expires_date": (datetime.now() + timedelta(days=random.randint(30, 365))).strftime("%Y-%m-%d"),
                "access_level": random.choice(["executive", "manager", "employee", "contractor"]),
                "encrypted": random.choice([True, False]),
                "backup_location": f"secure-server-{random.randint(1, 10)}.company.com"
            }
            company_secrets.append(secret)
        
        self.company_secrets = company_secrets
        return company_secrets
    
    def generate_proprietary_algorithms(self, num_records: int = 30) -> List[Dict]:
        """Generate synthetic proprietary algorithms"""
        algorithm_types = [
            "machine_learning", "encryption", "compression", "optimization",
            "pattern_recognition", "data_processing", "security_analysis", "prediction_model"
        ]
        
        programming_languages = ["Python", "Java", "C++", "JavaScript", "Go", "Rust", "Scala"]
        
        algorithms = []
        for i in range(num_records):
            algo_type = random.choice(algorithm_types)
            language = random.choice(programming_languages)
            
            # Generate realistic algorithm code
            algorithm_code = self._generate_algorithm_code(algo_type, language)
            
            algorithm = {
                "algorithm_id": f"ALG-{str(i+1).zfill(6)}",
                "name": f"{algo_type.replace('_', ' ').title()} Algorithm v{random.randint(1, 5)}.{random.randint(0, 9)}",
                "type": algo_type,
                "programming_language": language,
                "code": algorithm_code,
                "description": f"Proprietary {algo_type.replace('_', ' ')} algorithm for internal use",
                "version": f"{random.randint(1, 5)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
                "created_date": (datetime.now() - timedelta(days=random.randint(100, 1000))).strftime("%Y-%m-%d"),
                "last_updated": (datetime.now() - timedelta(days=random.randint(1, 100))).strftime("%Y-%m-%d"),
                "performance_metrics": {
                    "accuracy": round(random.uniform(0.85, 0.99), 3),
                    "speed": f"{random.randint(10, 1000)}ms",
                    "memory_usage": f"{random.randint(10, 500)}MB"
                },
                "dependencies": random.sample(["numpy", "pandas", "scikit-learn", "tensorflow", "pytorch"], random.randint(1, 3)),
                "license": "proprietary",
                "patent_pending": random.choice([True, False])
            }
            algorithms.append(algorithm)
        
        self.proprietary_algorithms = algorithms
        return algorithms
    
    def generate_security_protocols(self, num_records: int = 40) -> List[Dict]:
        """Generate synthetic security protocols"""
        protocol_types = [
            "authentication", "authorization", "encryption", "key_management",
            "access_control", "audit_logging", "incident_response", "backup_recovery"
        ]
        
        security_levels = ["low", "medium", "high", "critical"]
        
        protocols = []
        for i in range(num_records):
            protocol_type = random.choice(protocol_types)
            security_level = random.choice(security_levels)
            
            # Generate realistic protocol specification
            protocol_spec = self._generate_protocol_specification(protocol_type)
            
            protocol = {
                "protocol_id": f"SEC-{str(i+1).zfill(6)}",
                "name": f"{protocol_type.replace('_', ' ').title()} Protocol",
                "type": protocol_type,
                "security_level": security_level,
                "specification": protocol_spec,
                "implementation_date": (datetime.now() - timedelta(days=random.randint(30, 500))).strftime("%Y-%m-%d"),
                "last_reviewed": (datetime.now() - timedelta(days=random.randint(1, 90))).strftime("%Y-%m-%d"),
                "next_review": (datetime.now() + timedelta(days=random.randint(30, 365))).strftime("%Y-%m-%d"),
                "compliance": random.choice(["GDPR", "HIPAA", "SOX", "PCI-DSS", "ISO-27001"]),
                "encryption_standard": random.choice(["AES-256", "RSA-2048", "SHA-256", "ChaCha20"]),
                "key_rotation_days": random.choice([30, 60, 90, 180, 365]),
                "access_controls": {
                    "multi_factor_auth": random.choice([True, False]),
                    "role_based_access": random.choice([True, False]),
                    "ip_whitelisting": random.choice([True, False]),
                    "session_timeout": random.randint(15, 480)
                }
            }
            protocols.append(protocol)
        
        self.security_protocols = protocols
        return protocols
    
    def _generate_secret_content(self, secret_type: str) -> str:
        """Generate realistic secret content based on type"""
        if secret_type == "product_roadmap":
            return f"Q{random.randint(1, 4)} {random.randint(2024, 2026)}: Launch new AI-powered security platform with advanced threat detection capabilities. Budget: ${random.randint(500000, 5000000)}. Team size: {random.randint(10, 50)} engineers."
        elif secret_type == "acquisition_plan":
            return f"Planning to acquire {random.choice(['CyberTech Inc', 'SecureFlow Systems', 'DataGuard Solutions'])} for ${random.randint(10000000, 100000000)}. Expected closing date: {random.randint(1, 12)}/{random.randint(1, 28)}/{random.randint(2024, 2025)}."
        elif secret_type == "salary_data":
            return f"Engineering team average salary: ${random.randint(80000, 150000)}. Sales team average salary: ${random.randint(60000, 120000)}. Executive compensation package: ${random.randint(200000, 500000)} base + {random.randint(10, 50)}% bonus."
        else:
            return f"Confidential {secret_type.replace('_', ' ')} information. Access restricted to authorized personnel only. Document classification: {random.choice(['confidential', 'secret', 'top_secret'])}."
    
    def _generate_algorithm_code(self, algo_type: str, language: str) -> str:
        """Generate realistic algorithm code"""
        if language == "Python":
            if algo_type == "machine_learning":
                return '''def proprietary_ml_algorithm(data, parameters):
    """
    Proprietary machine learning algorithm for pattern recognition
    """
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    
    # Preprocessing
    processed_data = preprocess_input(data)
    
    # Feature extraction
    features = extract_features(processed_data)
    
    # Model training with proprietary parameters
    model = RandomForestClassifier(
        n_estimators=parameters.get('n_estimators', 100),
        max_depth=parameters.get('max_depth', 10),
        random_state=42
    )
    
    # Proprietary optimization
    optimized_model = apply_proprietary_optimization(model, features)
    
    return optimized_model'''
            else:
                return f'''def {algo_type}_algorithm(input_data):
    """
    Proprietary {algo_type.replace('_', ' ')} algorithm
    """
    # Implementation details are confidential
    result = process_data(input_data)
    return result'''
        else:
            return f"// Proprietary {algo_type.replace('_', ' ')} algorithm implementation in {language}\n// Code details are confidential"
    
    def _generate_protocol_specification(self, protocol_type: str) -> str:
        """Generate realistic protocol specification"""
        if protocol_type == "authentication":
            return f"""
1. User submits credentials via secure HTTPS connection
2. System validates credentials against encrypted database
3. Multi-factor authentication required for sensitive operations
4. Session tokens expire after {random.randint(15, 60)} minutes
5. Failed login attempts trigger account lockout after {random.randint(3, 10)} attempts
6. All authentication events logged for audit purposes
            """
        elif protocol_type == "encryption":
            return f"""
1. Data encrypted using {random.choice(['AES-256', 'ChaCha20'])} algorithm
2. Key rotation every {random.randint(30, 90)} days
3. Hardware security modules (HSM) for key storage
4. End-to-end encryption for all sensitive communications
5. Backup encryption keys stored in secure vault
            """
        else:
            return f"""
1. {protocol_type.replace('_', ' ').title()} protocol implementation
2. Security level: {random.choice(['low', 'medium', 'high', 'critical'])}
3. Compliance with {random.choice(['GDPR', 'HIPAA', 'SOX', 'PCI-DSS'])} standards
4. Regular security audits and penetration testing
5. Incident response procedures documented
            """
    
    def generate_all_data(self) -> Dict[str, List]:
        """Generate all types of synthetic data"""
        print("🔧 Generating synthetic customer data...")
        self.generate_customer_personal_data()
        
        print("💰 Generating synthetic financial records...")
        self.generate_financial_records()
        
        print("🤐 Generating synthetic company secrets...")
        self.generate_company_secrets()
        
        print("🧮 Generating synthetic proprietary algorithms...")
        self.generate_proprietary_algorithms()
        
        print("🔒 Generating synthetic security protocols...")
        self.generate_security_protocols()
        
        return {
            "customer_data": self.customer_data,
            "financial_records": self.financial_records,
            "company_secrets": self.company_secrets,
            "proprietary_algorithms": self.proprietary_algorithms,
            "security_protocols": self.security_protocols
        }
    
    def save_to_files(self, output_dir: str = "data"):
        """Save generated data to JSON files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        data_types = {
            "customer_data": self.customer_data,
            "financial_records": self.financial_records,
            "company_secrets": self.company_secrets,
            "proprietary_algorithms": self.proprietary_algorithms,
            "security_protocols": self.security_protocols
        }
        
        for data_type, data in data_types.items():
            filename = f"{output_dir}/{data_type}.json"
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"💾 Saved {len(data)} {data_type} records to {filename}")
        
        # Save combined dataset
        combined_data = {
            "metadata": {
                "generated_date": datetime.now().isoformat(),
                "total_records": sum(len(data) for data in data_types.values()),
                "data_types": list(data_types.keys())
            },
            "data": data_types
        }
        
        combined_filename = f"{output_dir}/combined_synthetic_data.json"
        with open(combined_filename, 'w') as f:
            json.dump(combined_data, f, indent=2)
        print(f"💾 Saved combined dataset to {combined_filename}")

def main():
    """Main function to generate synthetic data"""
    print("🤖 Synthetic Data Generator for AI Training")
    print("=" * 50)
    
    generator = SyntheticDataGenerator()
    
    # Generate all data
    all_data = generator.generate_all_data()
    
    # Save to files
    generator.save_to_files()
    
    # Print summary
    print("\n📊 Generated Data Summary:")
    print("-" * 30)
    for data_type, data in all_data.items():
        print(f"{data_type.replace('_', ' ').title()}: {len(data)} records")
    
    print(f"\n✅ Total records generated: {sum(len(data) for data in all_data.values())}")
    print("🎯 Data is ready for AI model training!")

if __name__ == "__main__":
    main() 