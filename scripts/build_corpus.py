import json
import time
from Bio import Entrez
from Bio import Medline

# Email needed by NCBI Entrez API

Entrez.email = "adrianjrosario8@gmail.com"

# Search queries related to TB CNS ADRs

SEARCH_TERMS = ["tuberculosis neurological adverse drug reaction",
                "TB CNS toxicity",
                "isoniazid peripheral neuropathy",
                "tuberculosis neurotoxicity",
                "anti-tubercular drug neurological side effects"]

# Max. no. of papers per search term

MAX_RESULTS = 50

# JSON file as output

OUTPUT_FILE = "../data/pubmed_tb_adr.json"

# Search PubMed

def search_pubmed(query, max_results=50):
    
    """
    Searches PUBMED and returns a list of PMIDs.
    
    """
    
    print(f"\nSearching PubMed for: {query}")
    
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax= max_results
    )
    
    results = Entrez.read(handle)
    
    pmid_list = results["IdList"]
    
    print(f"found {len(pmid_list)} papers.")
    
    return pmid_list

# Fetch the abstracts

def fetch_paper_details(pmid_list):
    
    """
    
    Fetches title, abstract, PMID from PubMed.
    
    """
    
    papers = []
    
    if not pmid_list:
        return papers
    
    ids = ",".join(pmid_list)   # IDS gets converted to string
    
    handle =  Entrez.efetch(
        db="pubmed",
        id=ids,
        rettype="medline",
        retmode="text"
    )
    
    records = Medline.parse(handle)
    
    for record in records:
        
        title = record.get("TI", "")
        abstract = record.get("AB", "")
        pmid = record.get("PMID", "")
        
        # Skip papers with missing abstracts
        
        if abstract.strip() == "":
            continue
        
        paper_data = {
            "pmid": pmid,
            "title": title,
            "abstract": abstract
        }
        
        papers.append(paper_data)
        
    return papers

# Main papers

def main():
    
    all_papers = []
    
    for query in SEARCH_TERMS:
        
        try:
            
            pmids = search_pubmed(query, MAX_RESULTS)
            
            papers = fetch_paper_details(pmids)
            
            all_papers.extend(papers)
            
            time.sleep(1)   # avoid over working the API
            
        except Exception as e:
            
            print(f"Error processing query '{query}': {e}")
            
            
    # Duplicate removal
    
    unique_papers = {}

    for paper in all_papers:

        unique_papers[paper["pmid"]] = paper

    final_papers = list(unique_papers.values())

    print(f"\nTotal unique papers collected: {len(final_papers)}")

    # -----------------------------
    # Save to JSON
    # -----------------------------

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        json.dump(final_papers, f, indent=4)

    print(f"\nCorpus saved to: {OUTPUT_FILE}")


# Run the script

if __name__ == "__main__":

    main()
    
    
            
            
     
    


        



