/**
 * Crop Health Monitoring System - Main Application
 * 
 * Handles map initialization, data visualization, and user interactions
 */

// Global variables
let map;
let ndviChart;
let currentData = null;

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    initMap();
    initChart();
    loadSampleData();
});

/**
 * Initialize Leaflet map
 */
function initMap() {
    // Create map centered on default location
    map = L.map('map').setView([12.9716, 77.5946], 13);
    
    // Add OpenStreetMap tile layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 19
    }).addTo(map);
    
    // Add sample field polygons (demonstration)
    addSampleFields();
}

/**
 * Add sample field markers/polygons for demonstration
 */
function addSampleFields() {
    // Sample field coordinates (mock data)
    const fields = [
        {
            name: 'Field A1',
            coords: [[12.9716, 77.5946], [12.9720, 77.5946], [12.9720, 77.5950], [12.9716, 77.5950]],
            ndvi: 0.75,
            health: 'Healthy'
        },
        {
            name: 'Field A2',
            coords: [[12.9710, 77.5946], [12.9714, 77.5946], [12.9714, 77.5950], [12.9710, 77.5950]],
            ndvi: 0.58,
            health: 'Mild Stress'
        },
        {
            name: 'Field A3',
            coords: [[12.9720, 77.5940], [12.9724, 77.5940], [12.9724, 77.5944], [12.9720, 77.5944]],
            ndvi: 0.38,
            health: 'Severe Stress'
        }
    ];
    
    fields.forEach(field => {
        const color = getHealthColor(field.ndvi);
        
        const polygon = L.polygon(field.coords, {
            color: color,
            fillColor: color,
            fillOpacity: 0.6
        }).addTo(map);
        
        polygon.bindPopup(`
            <strong>${field.name}</strong><br>
            NDVI: ${field.ndvi.toFixed(2)}<br>
            Health: <span style="color: ${color}">${field.health}</span><br>
            <a href="#" onclick="viewFieldDetails('${field.name}')">View Details</a>
        `);
    });
}

/**
 * Get color based on NDVI value
 */
function getHealthColor(ndvi) {
    if (ndvi > 0.65) return '#4caf50';  // Healthy
    if (ndvi > 0.45) return '#ffc107';  // Mild stress
    if (ndvi > 0.25) return '#ff9800';  // Severe stress
    return '#f44336';  // Critical
}

/**
 * Initialize NDVI time series chart
 */
function initChart() {
    const ctx = document.getElementById('ndviChart').getContext('2d');
    
    // Sample data (will be replaced with real data)
    const sampleDates = generateDateRange('2024-01-01', '2024-12-31', 30);
    const sampleNDVI = generateSampleNDVI(sampleDates.length);
    
    ndviChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: sampleDates,
            datasets: [{
                label: 'NDVI',
                data: sampleNDVI,
                borderColor: '#4caf50',
                backgroundColor: 'rgba(76, 175, 80, 0.1)',
                borderWidth: 3,
                tension: 0.4,
                pointRadius: 5,
                pointHoverRadius: 7
            }, {
                label: 'Critical Threshold',
                data: Array(sampleDates.length).fill(0.4),
                borderColor: '#f44336',
                borderWidth: 2,
                borderDash: [5, 5],
                pointRadius: 0,
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += context.parsed.y.toFixed(3);
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    min: 0,
                    max: 1,
                    title: {
                        display: true,
                        text: 'NDVI Value',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Date',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    },
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            }
        }
    });
}

/**
 * Generate date range for sample data
 */
function generateDateRange(start, end, numPoints) {
    const dates = [];
    const startDate = new Date(start);
    const endDate = new Date(end);
    const interval = (endDate - startDate) / (numPoints - 1);
    
    for (let i = 0; i < numPoints; i++) {
        const date = new Date(startDate.getTime() + interval * i);
        dates.push(date.toISOString().split('T')[0]);
    }
    
    return dates;
}

/**
 * Generate sample NDVI data with realistic patterns
 */
function generateSampleNDVI(numPoints) {
    const data = [];
    let baseValue = 0.5;
    
    for (let i = 0; i < numPoints; i++) {
        // Simulate growing season curve with some noise
        const seasonalEffect = 0.3 * Math.sin((i / numPoints) * Math.PI);
        const noise = (Math.random() - 0.5) * 0.1;
        const value = Math.max(0.2, Math.min(0.9, baseValue + seasonalEffect + noise));
        data.push(value);
    }
    
    return data;
}

/**
 * Load sample data (mock function - replace with API call)
 */
function loadSampleData() {
    // Simulate data loading
    currentData = {
        healthy: 65,
        mild: 25,
        severe: 8,
        critical: 2,
        currentNDVI: 0.72,
        trend: 'increasing',
        healthClass: 'Healthy',
        confidence: 87,
        yieldPotential: 92
    };
    
    updateDashboard(currentData);
}

/**
 * Update dashboard with new data
 */
function updateDashboard(data) {
    // Update health percentages
    document.getElementById('healthyPercent').textContent = data.healthy + '%';
    document.getElementById('mildPercent').textContent = data.mild + '%';
    document.getElementById('severePercent').textContent = data.severe + '%';
    document.getElementById('criticalPercent').textContent = data.critical + '%';
    
    // Update current status
    document.getElementById('currentNDVI').textContent = data.currentNDVI.toFixed(2);
    document.getElementById('trendStatus').textContent = getTrendIcon(data.trend) + ' ' + capitalize(data.trend);
    document.getElementById('healthClass').textContent = data.healthClass;
    document.getElementById('confidence').textContent = data.confidence + '%';
    document.getElementById('yieldPotential').textContent = data.yieldPotential + '%';
    
    // Update timestamp
    const now = new Date().toISOString().split('T')[0];
    document.getElementById('lastUpdate').textContent = `Last updated: ${now}`;
}

/**
 * Get trend icon
 */
function getTrendIcon(trend) {
    const icons = {
        'increasing': '↗️',
        'stable': '→',
        'decreasing': '↘️'
    };
    return icons[trend] || '→';
}

/**
 * Capitalize string
 */
function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

/**
 * Update location and reload data
 */
function updateLocation() {
    const lat = parseFloat(document.getElementById('latitude').value);
    const lon = parseFloat(document.getElementById('longitude').value);
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;
    
    if (isNaN(lat) || isNaN(lon)) {
        alert('Please enter valid coordinates');
        return;
    }
    
    // Center map on new location
    map.setView([lat, lon], 13);
    
    // Show loading indicator
    showLoading();
    
    // Simulate API call (replace with actual backend call)
    setTimeout(() => {
        hideLoading();
        alert(`Data loaded for coordinates: ${lat}, ${lon}\nDate range: ${startDate} to ${endDate}`);
        // In production: fetch data from backend API
        // fetchCropHealthData(lat, lon, startDate, endDate);
    }, 1000);
}

/**
 * View field details
 */
function viewFieldDetails(fieldName) {
    alert(`Viewing details for ${fieldName}\n\nThis would show:\n- Detailed NDVI history\n- AI classification results\n- Temporal trends\n- Recommended actions`);
    // In production: open modal or navigate to detailed view
}

/**
 * Export report as PDF
 */
function exportReport() {
    alert('Generating comprehensive crop health report...\n\nReport will include:\n- Field maps\n- Time series analysis\n- AI predictions\n- Recommendations\n\nIn production: Generate PDF using backend service');
    // In production: call backend API to generate PDF report
}

/**
 * Export data as CSV
 */
function exportData() {
    // Create sample CSV data
    const csvContent = `Date,NDVI,NDWI,Health_Classification,Confidence
2024-01-15,0.68,0.25,Healthy,0.89
2024-02-01,0.72,0.23,Healthy,0.91
2024-02-15,0.65,0.28,Healthy,0.85`;
    
    // Create download link
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'crop_health_data.csv';
    a.click();
    window.URL.revokeObjectURL(url);
}

/**
 * Share analysis link
 */
function shareLink() {
    const url = window.location.href;
    navigator.clipboard.writeText(url).then(() => {
        alert('Analysis link copied to clipboard!');
    }).catch(err => {
        alert('Link: ' + url);
    });
}

/**
 * Show about modal
 */
function showAbout() {
    alert(`AI Crop Health Monitoring System

Version: 1.0 (Research Prototype)

Description:
This system integrates Sentinel-2 satellite imagery with machine learning 
to provide automated crop health classification, early stress detection, 
and relative yield potential estimation.

Key Features:
✓ Multi-temporal NDVI analysis
✓ Random Forest health classification
✓ Early warning system
✓ Relative yield estimation
✓ Interactive web GIS interface

Technology Stack:
- Sentinel-2 Level-2A imagery
- Python (scikit-learn, rasterio, pandas)
- Leaflet.js for mapping
- Chart.js for visualization

Disclaimer:
This is a research and educational tool. Results should be validated 
with field observations and agronomic expertise.

Contact: Agricultural AI Research Project`);
}

/**
 * Show help information
 */
function showHelp() {
    alert(`Help - How to Use This System

1. SELECT LOCATION:
   - Enter latitude and longitude coordinates
   - Or click on the map to select a point
   - Choose date range for analysis

2. INTERPRET RESULTS:
   - Green (Healthy): NDVI > 0.65
   - Yellow (Mild Stress): NDVI 0.45-0.65
   - Orange (Severe Stress): NDVI 0.25-0.45
   - Red (Critical): NDVI < 0.25

3. MONITOR TRENDS:
   - Check the time series chart for patterns
   - Look for declining trends (early warning)
   - Note anomalies marked on the chart

4. REVIEW ALERTS:
   - Critical alerts require immediate attention
   - Warning alerts suggest investigation
   - Watch status indicates increased monitoring

5. EXPORT DATA:
   - Download full reports as PDF
   - Export raw data as CSV for analysis
   - Share analysis with team members

IMPORTANT:
Always validate satellite-based insights with field observations!`);
}

/**
 * Show loading indicator
 */
function showLoading() {
    // Simple loading implementation
    document.body.style.cursor = 'wait';
}

/**
 * Hide loading indicator
 */
function hideLoading() {
    document.body.style.cursor = 'default';
}

/**
 * Fetch crop health data from backend API (placeholder)
 * In production, this would call your Flask/FastAPI backend
 */
async function fetchCropHealthData(lat, lon, startDate, endDate) {
    try {
        const response = await fetch(`/api/crop-health?lat=${lat}&lon=${lon}&start=${startDate}&end=${endDate}`);
        
        if (!response.ok) {
            throw new Error('Failed to fetch data');
        }
        
        const data = await response.json();
        
        // Update dashboard with real data
        updateDashboard(data.summary);
        
        // Update chart with real time series
        updateChart(data.timeseries);
        
        // Update map with field polygons
        updateMap(data.fields);
        
        // Update alerts
        updateAlerts(data.alerts);
        
    } catch (error) {
        console.error('Error fetching crop health data:', error);
        alert('Error loading data. Please try again.');
    }
}

/**
 * Update chart with new data
 */
function updateChart(timeseries) {
    if (!timeseries || timeseries.length === 0) return;
    
    const dates = timeseries.map(d => d.date);
    const ndviValues = timeseries.map(d => d.ndvi);
    
    ndviChart.data.labels = dates;
    ndviChart.data.datasets[0].data = ndviValues;
    ndviChart.update();
}

/**
 * Update map with field data
 */
function updateMap(fields) {
    // Clear existing layers (except base tile layer)
    map.eachLayer(layer => {
        if (layer instanceof L.Polygon) {
            map.removeLayer(layer);
        }
    });
    
    // Add new field polygons
    fields.forEach(field => {
        const color = getHealthColor(field.ndvi);
        
        const polygon = L.polygon(field.coordinates, {
            color: color,
            fillColor: color,
            fillOpacity: 0.6
        }).addTo(map);
        
        polygon.bindPopup(`
            <strong>${field.name}</strong><br>
            NDVI: ${field.ndvi.toFixed(2)}<br>
            Health: ${field.health}<br>
            Confidence: ${field.confidence}%
        `);
    });
}

/**
 * Update alerts panel
 */
function updateAlerts(alerts) {
    const container = document.getElementById('alertContainer');
    container.innerHTML = '';
    
    if (!alerts || alerts.length === 0) {
        container.innerHTML = '<p class="text-muted">No active alerts</p>';
        return;
    }
    
    alerts.forEach(alert => {
        const alertClass = `alert-${alert.severity}`;
        const icon = alert.severity === 'critical' ? '🔴' : 
                    alert.severity === 'warning' ? '⚠️' : '👁️';
        
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert-item ${alertClass}`;
        alertDiv.innerHTML = `
            <strong>${icon} ${alert.title}</strong>
            <p class="mb-1">${alert.message}</p>
            <small class="text-muted">Detected: ${alert.date}</small>
        `;
        
        container.appendChild(alertDiv);
    });
}

// Console message
console.log(`
╔═══════════════════════════════════════════════════════════╗
║   AI Crop Health Monitoring System                       ║
║   Research Prototype v1.0                                 ║
║                                                           ║
║   Integrating satellite imagery, machine learning,       ║
║   and web GIS for agricultural remote sensing            ║
╚═══════════════════════════════════════════════════════════╝
`);
