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

            # Step 3: Skip individual analysis, go directly to combined analysis
            logging.info("🎯 Running combined analysis (skipping individual analysis to avoid duplicates)...")
            final_success = self.generate_final_combined_analysis()

            if final_success:
                logging.info("🎉 Complete pipeline finished successfully!")
                
                # Show summary
                data_files = self.get_latest_data_files()
                analyzed_files = glob.glob("analysis_results/*_analyzed_*.csv")

                logging.info(f"\n📊 PIPELINE SUMMARY:")
                logging.info(f" Raw data files: {len(data_files)}")
                logging.info(f" Final analyzed file: 1 (combined analysis)")
                logging.info(f" Total runtime: {datetime.now() - self.start_time}")
                logging.info(f" Output folder: analysis_results/")

                return True
            else:
                logging.error("❌ Combined analysis failed")
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
        
    
    def generate_final_combined_analysis(self):
        """Generate a single final output file combining all collections"""
        try:
            # Import the combined analysis function
            import sys
            sys.path.append('.')
            from reddit_data_analyzer import analyze_combined_reddit_data
            
            # Find all raw data files (not analyzed ones)
            import glob
            all_data_files = []
            
            # Search for different collection types
            daily_files = glob.glob("analysis_results/*daily*.csv") + glob.glob("analysis_results/*activity_based*.csv")
            search_files = glob.glob("analysis_results/*search*.csv") + glob.glob("analysis_results/*data_tools*.csv")
            
            # Filter out already analyzed files
            daily_files = [f for f in daily_files if '_analyzed_' not in f]
            search_files = [f for f in search_files if '_analyzed_' not in f]
            
            all_data_files = daily_files + search_files
            
            if not all_data_files:
                logging.warning("⚠️  No raw data files found for combined analysis")
                return False
            
            logging.info(f"🔍 Found {len(all_data_files)} files to combine:")
            for file in all_data_files:
                logging.info(f"   - {os.path.basename(file)}")
            
            # Set up analysis parameters
            engagement_weights = {
                'score': 0.3,
                'upvote_ratio': 0.25,
                'comments': 0.25,
                'comments_quality': 0.15,
                'recency': 0.05
            }
            
            filter_thresholds = {
                'min_engagement_score': 45,  # Stricter quality threshold
                'min_score': 10,              # Require meaningful engagement
                'min_comments': 5,            # Require actual discussion
                'min_upvote_ratio': 0.7,      # Higher community approval
                'max_days_ago': 7             # Focus on recent, relevant content
            }
            
            # Run combined analysis
            logging.info("🚀 Running combined analysis...")
            result = analyze_combined_reddit_data(
                file_paths=all_data_files,
                engagement_weights=engagement_weights,
                filter_thresholds=filter_thresholds,
                save_unfiltered=False
            )
            
            if result is not None:
                logging.info("✅ Final combined analysis complete")
                logging.info("📄 Single output file created with all data combined and analyzed")
                return True
            else:
                logging.error("❌ Combined analysis failed")
                return False
                
        except Exception as e:
            logging.error(f"❌ Failed to generate combined analysis: {e}")
            return False

def main():
    """Main orchestration function"""
    orchestrator = RedditOrchestrator()
    
    # Configuration - can be overridden by command line argument
    default_collection_type = "both"  # Default to both daily and search
    
    # Check for command line argument
    if len(sys.argv) > 1:
        collection_arg = sys.argv[1].lower()
        if collection_arg in ["daily", "search", "both"]:
            COLLECTION_TYPE = collection_arg
        elif collection_arg in ["-h", "--help", "help"]:
            print("🤖 Reddit Data Orchestrator - Usage:")
            print("python reddit_orchestrator.py [daily|search|both]")
            print()
            print("Collection types:")
            print("  daily  - Run only daily subreddit collection")
            print("  search - Run only keyword-based search collection") 
            print("  both   - Run both daily and search collection (default)")
            print()
            print("Examples:")
            print("  python reddit_orchestrator.py")         # both (default)
            print("  python reddit_orchestrator.py search")  # search only
            print("  python reddit_orchestrator.py daily")   # daily only
            sys.exit(0)
        else:
            print(f"❌ Invalid collection type: {collection_arg}")
            print("Valid options: daily, search, both")
            print("Use --help for more information")
            sys.exit(1)
    else:
        COLLECTION_TYPE = default_collection_type
    
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