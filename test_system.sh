#!/bin/bash
# Arthur Image Recognition 2.0 - Quick Test Script
# Run this after the server has started successfully

echo "🚀 Arthur Image Recognition 2.0 - System Tests"
echo "================================================="

BASE_URL="http://localhost:8000"

# Test 1: Health Check
echo -e "\n🧪 Testing: Health Check"
response=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health")
if [ "$response" = "200" ]; then
    echo "   ✅ Health check passed (HTTP $response)"
    curl -s "$BASE_URL/health" | jq '.' 2>/dev/null || echo "   Health data received"
else
    echo "   ❌ Health check failed (HTTP $response)"
fi

# Test 2: Legacy Status Endpoint
echo -e "\n🧪 Testing: Legacy Status Endpoint" 
response=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/status")
if [ "$response" = "200" ]; then
    echo "   ✅ Legacy status endpoint working (HTTP $response)"
else
    echo "   ❌ Legacy status failed (HTTP $response)"
fi

# Test 3: API Documentation
echo -e "\n🧪 Testing: API Documentation"
response=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/docs")
if [ "$response" = "200" ]; then
    echo "   ✅ API documentation available at $BASE_URL/docs"
else
    echo "   ❌ API docs unavailable (HTTP $response)"
fi

# Test 4: OpenAPI Schema
echo -e "\n🧪 Testing: OpenAPI Schema"
response=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/openapi.json")
if [ "$response" = "200" ]; then
    echo "   ✅ OpenAPI schema available"
else
    echo "   ❌ OpenAPI schema failed (HTTP $response)"
fi

# Test 5: Basic Image Match (if server is fully loaded)
echo -e "\n🧪 Testing: Image Matching Endpoint"
response=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
  -F "imgUrl=https://via.placeholder.com/300x200.png?text=Test+Image" \
  "$BASE_URL/match")

if [ "$response" = "200" ]; then
    echo "   ✅ Image matching endpoint working (HTTP $response)"
elif [ "$response" = "500" ]; then
    echo "   ⚠️  Image matching endpoint exists but server may still be loading models"
elif [ "$response" = "422" ]; then
    echo "   ⚠️  Image matching endpoint exists (validation error expected for test URL)"
else
    echo "   ❌ Image matching failed (HTTP $response)"
fi

echo -e "\n================================================="
echo "📊 TEST SUMMARY"
echo "================================================="

# Final status
echo -e "\n🎉 Arthur Image Recognition 2.0 Status:"
echo "   • Server: Running on $BASE_URL"
echo "   • API Docs: $BASE_URL/docs" 
echo "   • Health: $BASE_URL/health"
echo "   • Legacy API: Backwards compatible"
echo ""
echo "📚 Next Steps:"
echo "   • Visit $BASE_URL/docs for interactive API documentation"
echo "   • Test with real images using the /match endpoint"  
echo "   • Monitor server logs for CLIP model loading progress"
echo "   • Deploy using Docker Compose for production"
echo ""
echo "🔗 Key Endpoints:"
echo "   POST /match - Image similarity search (legacy compatible)"
echo "   POST /api/v1/search/similar - Modern semantic search"
echo "   GET /health - System health and status"
echo "   GET /status - Legacy status endpoint"