# Dashboard UI Upgrade

## Theme System

The dashboard uses CSS variables for theming, with three built-in themes:

- **Light**: Clean, professional look with blue accents
- **Dark**: Eye-friendly dark mode with cyan accents
- **Clinic**: Medical-themed with teal accents and subtle shadows

Theme variables are defined in `client/styles.css` and applied via `data-theme` attribute on the HTML root.

## Port Auto-detection

The API client automatically detects the correct endpoint:

```javascript
const DEFAULT_API = `${window.location.protocol}//${window.location.hostname}:7860`;
const API_BASE = window.SURGICALAI_API_BASE || DEFAULT_API;
```

This allows the dashboard to work seamlessly whether:
- Served directly from the FastAPI server (port 7860)
- Served from a development server (port 8080)
- Behind a reverse proxy (override with window.SURGICALAI_API_BASE)

## Form Improvements

- Added comprehensive facial site options based on common surgical areas
- Included standard diagnosis options from dermatological practice
- Clear dropzone UI with file validation and preview
- Proper labeling and aria attributes for accessibility

## Progress & Feedback

- Streaming mode shows real-time analysis progress
- Non-streaming mode shows determinate progress bar
- Toast notifications for key events
- Clear error states with request IDs for debugging

## Quick Start

1. Start the server:
   ```powershell
   python -m uvicorn server.http_api:app --port 7860
   ```

2. Open http://localhost:7860 in your browser

3. Try the theme switcher in the top-right

4. Enable streaming mode in Settings for real-time analysis

## Logs & Debugging

- Usage logs available in the side drawer (clock icon)
- Each request gets a unique ID shown in the footer
- Network errors show detailed information and request IDs
- Theme changes persist in localStorage

## Production Considerations

- CSS variables support high-contrast mode
- Mobile-responsive with proper touch targets
- Reduced motion support for animations
- Clear error states and offline indicators
