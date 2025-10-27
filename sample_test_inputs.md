# Sample Test Inputs - Plush For You API

This file contains sample API test inputs you can use to test the recommendation system.

---

## Prerequisites

Make sure the API server is running:
```bash
python -m uvicorn src.serve:app --host 127.0.0.1 --port 8000
```

---

## 1. Health Check & System Stats

### PowerShell
```powershell
# Health check
Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" -UseBasicParsing | Select-Object -ExpandProperty Content

# System statistics
Invoke-WebRequest -Uri "http://127.0.0.1:8000/stats" -UseBasicParsing | Select-Object -ExpandProperty Content
```

### cURL
```bash
# Health check
curl "http://127.0.0.1:8000/health"

# System statistics
curl "http://127.0.0.1:8000/stats"
```

---

## 2. Cold-Start User (No History)

### PowerShell
```powershell
# Get 5 recommendations for a new user
Invoke-WebRequest -Uri "http://127.0.0.1:8000/for_you?user_id=brand_new_user&k=5" -UseBasicParsing | Select-Object -ExpandProperty Content

# Get 10 recommendations
Invoke-WebRequest -Uri "http://127.0.0.1:8000/for_you?user_id=cold_start_test&k=10" -UseBasicParsing | Select-Object -ExpandProperty Content

# Get 20 recommendations
Invoke-WebRequest -Uri "http://127.0.0.1:8000/for_you?user_id=new_user_2025&k=20" -UseBasicParsing | Select-Object -ExpandProperty Content
```

### cURL
```bash
curl "http://127.0.0.1:8000/for_you?user_id=brand_new_user&k=5"
curl "http://127.0.0.1:8000/for_you?user_id=cold_start_test&k=10"
curl "http://127.0.0.1:8000/for_you?user_id=new_user_2025&k=20"
```

**Expected**: Diverse brands, wide price range, exploratory recommendations

---

## 3. Existing Users (Different Profiles)

### User A: Heavy User (37 interactions)
**User ID**: `84c94984-a83d-429b-90a0-b07ac6989deb`  
**Profile**: Mixed brand preferences, $1,584 median price

```powershell
# Check user profile
Invoke-WebRequest -Uri "http://127.0.0.1:8000/debug/user/84c94984-a83d-429b-90a0-b07ac6989deb" -UseBasicParsing | Select-Object -ExpandProperty Content

# Get recommendations
Invoke-WebRequest -Uri "http://127.0.0.1:8000/for_you?user_id=84c94984-a83d-429b-90a0-b07ac6989deb&k=10" -UseBasicParsing | Select-Object -ExpandProperty Content
```

```bash
# cURL version
curl "http://127.0.0.1:8000/debug/user/84c94984-a83d-429b-90a0-b07ac6989deb"
curl "http://127.0.0.1:8000/for_you?user_id=84c94984-a83d-429b-90a0-b07ac6989deb&k=10"
```

**Expected**: Balanced recommendations, multiple brands, ~$1,584 median price

---

### User B: Costarellos Enthusiast (20 interactions, 42.8% affinity)
**User ID**: `01899331-38b2-4d3e-9641-c547d202ba0f`  
**Profile**: Strong Costarellos preference, $1,043 median price

```powershell
# Check user profile
Invoke-WebRequest -Uri "http://127.0.0.1:8000/debug/user/01899331-38b2-4d3e-9641-c547d202ba0f" -UseBasicParsing | Select-Object -ExpandProperty Content

# Get recommendations
Invoke-WebRequest -Uri "http://127.0.0.1:8000/for_you?user_id=01899331-38b2-4d3e-9641-c547d202ba0f&k=10" -UseBasicParsing | Select-Object -ExpandProperty Content
```

```bash
# cURL version
curl "http://127.0.0.1:8000/debug/user/01899331-38b2-4d3e-9641-c547d202ba0f"
curl "http://127.0.0.1:8000/for_you?user_id=01899331-38b2-4d3e-9641-c547d202ba0f&k=10"
```

**Expected**: High percentage of Costarellos items, prices around $1,043

---

### User C: Christopher Esber Superfan (16 interactions, 31.3% affinity)
**User ID**: `fa3364d4-b5de-4085-96f8-5bbdfea8e36a`  
**Profile**: Strong Christopher Esber preference, $952.50 median price

```powershell
# Check user profile
Invoke-WebRequest -Uri "http://127.0.0.1:8000/debug/user/fa3364d4-b5de-4085-96f8-5bbdfea8e36a" -UseBasicParsing | Select-Object -ExpandProperty Content

# Get recommendations
Invoke-WebRequest -Uri "http://127.0.0.1:8000/for_you?user_id=fa3364d4-b5de-4085-96f8-5bbdfea8e36a&k=10" -UseBasicParsing | Select-Object -ExpandProperty Content
```

```bash
# cURL version
curl "http://127.0.0.1:8000/debug/user/fa3364d4-b5de-4085-96f8-5bbdfea8e36a"
curl "http://127.0.0.1:8000/for_you?user_id=fa3364d4-b5de-4085-96f8-5bbdfea8e36a&k=10"
```

**Expected**: Nearly 100% Christopher Esber items, prices $420-$960, mostly black

---

### User D: Medium Activity (5 interactions)
**User ID**: `50cc3929-7d1b-4eab-ab3b-b6ab279afae0`

```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:8000/for_you?user_id=50cc3929-7d1b-4eab-ab3b-b6ab279afae0&k=10" -UseBasicParsing | Select-Object -ExpandProperty Content
```

```bash
curl "http://127.0.0.1:8000/for_you?user_id=50cc3929-7d1b-4eab-ab3b-b6ab279afae0&k=10"
```

---

### User E: Light User (1-2 interactions)
**User ID**: `9c72bb01-ac82-4234-8f43-edf682b58da7`

```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:8000/for_you?user_id=9c72bb01-ac82-4234-8f43-edf682b58da7&k=10" -UseBasicParsing | Select-Object -ExpandProperty Content
```

```bash
curl "http://127.0.0.1:8000/for_you?user_id=9c72bb01-ac82-4234-8f43-edf682b58da7&k=10"
```

**Expected**: Content-heavy weighting (sparse data), some personalization

---

## 4. Different Recommendation Counts

### PowerShell
```powershell
# Top 5 recommendations
Invoke-WebRequest -Uri "http://127.0.0.1:8000/for_you?user_id=fa3364d4-b5de-4085-96f8-5bbdfea8e36a&k=5" -UseBasicParsing | Select-Object -ExpandProperty Content

# Top 10 recommendations
Invoke-WebRequest -Uri "http://127.0.0.1:8000/for_you?user_id=fa3364d4-b5de-4085-96f8-5bbdfea8e36a&k=10" -UseBasicParsing | Select-Object -ExpandProperty Content

# Top 20 recommendations (default)
Invoke-WebRequest -Uri "http://127.0.0.1:8000/for_you?user_id=fa3364d4-b5de-4085-96f8-5bbdfea8e36a&k=20" -UseBasicParsing | Select-Object -ExpandProperty Content

# Top 50 recommendations
Invoke-WebRequest -Uri "http://127.0.0.1:8000/for_you?user_id=fa3364d4-b5de-4085-96f8-5bbdfea8e36a&k=50" -UseBasicParsing | Select-Object -ExpandProperty Content
```

### cURL
```bash
curl "http://127.0.0.1:8000/for_you?user_id=fa3364d4-b5de-4085-96f8-5bbdfea8e36a&k=5"
curl "http://127.0.0.1:8000/for_you?user_id=fa3364d4-b5de-4085-96f8-5bbdfea8e36a&k=10"
curl "http://127.0.0.1:8000/for_you?user_id=fa3364d4-b5de-4085-96f8-5bbdfea8e36a&k=20"
curl "http://127.0.0.1:8000/for_you?user_id=fa3364d4-b5de-4085-96f8-5bbdfea8e36a&k=50"
```

---

## 5. All Available User IDs (Top 10 by interaction count)

You can test with any of these real user IDs:

```
1. 84c94984-a83d-429b-90a0-b07ac6989deb  (37 interactions - Heavy user)
2. 01899331-38b2-4d3e-9641-c547d202ba0f  (20 interactions - Costarellos fan)
3. fa3364d4-b5de-4085-96f8-5bbdfea8e36a  (16 interactions - Esber superfan)
4. 50cc3929-7d1b-4eab-ab3b-b6ab279afae0  (5 interactions)
5. 3c374015-b2bd-40c2-9dc0-460092946770  (5 interactions)
6. f1ee63aa-7b5a-4d46-8936-9a6bbe97510f  (4 interactions)
7. 7321936b-a55f-46e9-b8ba-0c6c8b3d6e84  (4 interactions)
8. 9563ae84-05ae-4b52-843c-756a5d4e4e39  (3 interactions)
9. 997b08ce-5079-403a-9657-cd3e4c3b2290  (3 interactions)
10. 17cfaa97-c465-45ee-befc-a0e44fe213e3 (2 interactions)
```

### Quick Test
```powershell
# Test any user from the list above
$userId = "84c94984-a83d-429b-90a0-b07ac6989deb"
Invoke-WebRequest -Uri "http://127.0.0.1:8000/for_you?user_id=$userId&k=10" -UseBasicParsing | Select-Object -ExpandProperty Content
```

---

## 6. Interactive API Documentation

Open in your browser:
```
http://127.0.0.1:8000/docs
```

This provides a **Swagger UI** where you can:
- Test all endpoints interactively
- See request/response schemas
- Try different parameters
- View example responses

---

## 7. Python Test Script

Save as `test_api.py` and run with `python test_api.py`:

```python
import requests
import json

API_BASE = "http://127.0.0.1:8000"

# Test users
users = {
    "Cold-Start": "new_user_123",
    "Heavy User": "84c94984-a83d-429b-90a0-b07ac6989deb",
    "Esber Superfan": "fa3364d4-b5de-4085-96f8-5bbdfea8e36a"
}

for user_type, user_id in users.items():
    print(f"\n{'='*60}")
    print(f"{user_type}: {user_id}")
    print('='*60)
    
    # Get recommendations
    response = requests.get(f"{API_BASE}/for_you", params={"user_id": user_id, "k": 5})
    data = response.json()
    
    print(f"\nTop 5 Recommendations:")
    for i, item in enumerate(data['items'][:5], 1):
        print(f"{i}. {item['name']}")
        print(f"   Brand: {item['brand']} | Price: ${item['price']:.0f} | Color: {item['color']}")
```

---

## Expected Results

### Cold-Start User
- **Brands**: 5 different brands (maximum diversity)
- **Price Range**: Wide range ($360 - $5,000+)
- **Strategy**: Exploration

### Heavy User (37 interactions)
- **Brands**: 5 different brands (balanced)
- **Price Range**: Around $1,584 median
- **Strategy**: Balanced personalization

### Christopher Esber Superfan (31.3% affinity)
- **Brands**: 100% Christopher Esber
- **Price Range**: $420 - $960 (median $952.50)
- **Color**: Mostly black
- **Strategy**: Ultra-focused personalization

---

**All test inputs verified and working!**

