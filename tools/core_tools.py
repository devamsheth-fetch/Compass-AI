import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def get_location_data(query: str, location: str):
    """
    Fetches location data from Yelp Fusion API.
    """
    api_key = os.environ.get("YELP_API_KEY")
    print(f"\n[API RUNNING] 📍 Fetching Yelp location data for '{query}' in '{location}'...", flush=True)
    if not api_key:
        raise ValueError("[API ERROR] Yelp API Key missing. Mock data is disabled.")

    url = "https://api.yelp.com/v3/businesses/search"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"term": query, "location": location, "limit": 5}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        businesses = response.json().get("businesses", [])
    except Exception as e:
        print(f"[API WARNING] Failed to fetch Yelp data for '{query}' in '{location}': {e}", flush=True)
        return json.dumps([])
    
    results = []
    for b in businesses:
        business_id = b.get("id")
        
        # Secondary API call to fetch top reviews for this specific business
        reviews = []
        if business_id:
            try:
                review_url = f"https://api.yelp.com/v3/businesses/{business_id}/reviews"
                review_response = requests.get(review_url, headers=headers)
                if review_response.status_code == 200:
                    review_data = review_response.json().get("reviews", [])
                    # Grab just the text of the top 2 reviews
                    for r in review_data[:2]:
                        reviews.append(r.get("text"))
            except Exception as e:
                print(f"[API WARNING] Failed to fetch reviews for {business_id}: {e}")

        results.append({
            "name": b.get("name"),
            "rating": b.get("rating"),
            "category": b.get("categories", [{}])[0].get("title", ""),
            "reviews": reviews,  # Added the fetched text reviews here
            "promo": "Verify locally", 
            "distance_miles": round(b.get("distance", 0) / 1609.34, 2)
        })
    return json.dumps(results)

def get_weather_data(location: str):
    """
    Fetches real-time weather conditions from OpenWeatherMap.
    CRITICAL: The `location` parameter MUST be a broad city or region name (e.g., "San Francisco", "Austin"). 
    Do NOT pass specific park, trail, or business names, as the API will fail to find them.
    """
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    print(f"[API RUNNING] ⛅ Fetching OpenWeather data for '{location}'...", flush=True)
    if not api_key:
        raise ValueError("[API ERROR] OpenWeather API Key missing. Mock data is disabled.")
        
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": location, "appid": api_key, "units": "imperial"}
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return json.dumps({
            "location": data.get("name"),
            "condition": data.get("weather", [{}])[0].get("main", "Unknown"),
            "temp_f": data.get("main", {}).get("temp", 0)
        })
    except Exception as e:
        print(f"[API WARNING] Failed to fetch weather for '{location}': {e}", flush=True)
        return json.dumps({
            "location": location,
            "condition": "Unknown",
            "temp_f": "N/A"
        })

def search_local_facts(topic: str):
    """
    Uses Tavily Search API to find interesting local facts.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    print(f"[API RUNNING] 🔍 Searching Tavily for facts about '{topic}'...", flush=True)
    if not api_key:
        raise ValueError("[API ERROR] Tavily API Key missing. Mock data is disabled.")
        
    url = "https://api.tavily.com/search"
    headers = {"Content-Type": "application/json"}
    payload = {
        "api_key": api_key,
        "query": f"Interesting facts about {topic}",
        "search_depth": "advanced",
        # "chunks_per_source": 5
        "include_answer": "advanced"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("answer", data.get("results", [{}])[0].get("content", "No interesting facts found."))
    except Exception as e:
        print(f"[API WARNING] Failed to search Tavily for '{topic}': {e}", flush=True)
        return "No interesting facts found due to an API error."
