#!/usr/bin/env python3
"""
Reddit Data Collection and Analysis Orchestrator
Automatically runs data collection followed by analysis
"""

import subprocess
import sys
import time
import logging
import os
from datetime import datetime
import glob

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reddit_orchestrator.log'),
        logging.StreamHandler()
    ],
    force=True
)

class RedditOrchestrator:
    def __init__(self):
        """Initialize the orchestrator"""
        self.start_time = datetime.now()
        logging.info("Reddit Data Orchestrator started")
    
    def run_script(self, script_name, description):
        """Run a Python script and return success status with live output"""
        try:
            logging.info(f"Starting {description}...")
            logging.info(f"Running: python3 {script_name}")
            logging.info("-" * 60)
            
            # Run the script with live output streaming
            process = subprocess.Popen(
                [sys.executable, script_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    # Print to console and log
                    print(output.strip())
                    logging.info(f"[{script_name}] {output.strip()}")
            
            # Wait for process to complete
            return_code = process.poll()
            
            logging.info("-" * 60)
            if return_code == 0:
                logging.info(f"✅ {description} completed successfully")
                return True
            else:
                logging.error(f"❌ {description} failed with return code {return_code}")
                return False
                
        except Exception as e:
            logging.error(f"❌ {description} failed with exception: {e}")
            return False
    
    def get_latest_data_files(self):
        """Get the most recently created data files"""
        if not os.path.exists("analysis_results"):
            logging.warning("No analysis_results folder found")
            return []
        
        # Look for data files (not analyzed files)
        pattern_files = glob.glob("analysis_results/*.csv") + glob.glob("analysis_results/*.json")
        data_files = [f for f in pattern_files if not '_analyzed_' in f and not 'progress_' in f]
        
        if data_files:
            # Sort by creation time, most recent first
            data_files.sort(key=os.path.getctime, reverse=True)
            logging.info(f"Found {len(data_files)} data files for analysis")
            return data_files
        else:
            logging.warning("No data files found for analysis")
            return []
    
    def run_collection_and_analysis(self, collection_type="daily"):
        """
        Run data collection followed by analysis
        
        Args:
            collection_type (str): "daily", "search", or "both"
        """
        logging.info(f"🚀 Starting Reddit data collection and analysis pipeline")
        logging.info(f"Collection type: {collection_type}")
        
        collection_success = False
        
        # Step 1: Data Collection
        if collection_type in ["daily", "both"]:
            success = self.run_script(
                "reddit_daily_collector.py", 
                "Daily Reddit Data Collection"
            )
            if success:
                collection_success = True
        
        if collection_type in ["search", "both"]:
            success = self.run_script(
                "reddit_search_collector.py",
                "Reddit Search Data Collection" 
            )
            if success:
                collection_success = True
        
        # Step 2: Wait a moment for files to be written
        if collection_success:
            logging.info("⏳ Waiting 5 seconds for files to be written...")
            time.sleep(5)
            
            # Step 3: Data Analysis
            success = self.run_script(
                "reddit_data_analyzer.py",
                "Reddit Data Analysis"
            )
            
            if success:
                logging.info("🎉 Complete pipeline finished successfully!")
                
                # Show summary
                data_files = self.get_latest_data_files()
                analyzed_files = glob.glob("analysis_results/*_analyzed_*.csv")
                
                logging.info(f"\n📊 PIPELINE SUMMARY:")
                logging.info(f"   Raw data files: {len(data_files)}")
                logging.info(f"   Analyzed files: {len(analyzed_files)}")
                logging.info(f"   Total runtime: {datetime.now() - self.start_time}")
                logging.info(f"   Output folder: analysis_results/")
                
                return True
            else:
                logging.error("❌ Analysis step failed")
                return False
        else:
            logging.error("❌ Data collection failed - skipping analysis")
            return False
    
    def run_analysis_only(self):
        """Run analysis on existing data files"""
        logging.info("🔍 Running analysis on existing data...")
        
        data_files = self.get_latest_data_files()
        if not data_files:
            logging.error("❌ No data files found to analyze")
            return False
        
        success = self.run_script(
            "reddit_data_analyzer.py",
            "Reddit Data Analysis (existing data)"
        )
        
        if success:
            logging.info("✅ Analysis completed on existing data")
            return True
        else:
            logging.error("❌ Analysis failed")
            return False

def main():
    """Main orchestration function"""
    orchestrator = RedditOrchestrator()
    
    # Configuration
    COLLECTION_TYPE = "daily"  # Options: "daily", "search", "both"
    
    print("🤖 Reddit Data Collection & Analysis Orchestrator")
    print("=" * 50)
    print(f"Collection type: {COLLECTION_TYPE}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Run the complete pipeline
        success = orchestrator.run_collection_and_analysis(COLLECTION_TYPE)
        
        if success:
            print("\n🎉 Pipeline completed successfully!")
            print("📁 Check the analysis_results/ folder for output files")
        else:
            print("\n❌ Pipeline failed - check logs for details")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logging.info("⏹️  Pipeline interrupted by user")
        print("\n⏹️  Pipeline stopped by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"💥 Pipeline failed with unexpected error: {e}")
        print(f"\n💥 Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()