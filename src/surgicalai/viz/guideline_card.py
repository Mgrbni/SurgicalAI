"""Visualization for guideline summary cards."""

from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Color constants
COLORS = {
    "ink": "#0B0F14",
    "gold": "#D4AF37",
    "ice": "#E7EEF5",
    "alert": "#E4572E",
    "safe": "#2E7D32",
}

class GuidelineCard:
    """Renderer for guideline summary cards."""
    
    def __init__(self, size=(1080, 1350)):
        """Initialize with card dimensions."""
        self.width, self.height = size
        
        # Try to load fonts
        try:
            self.title_font = ImageFont.truetype("Spectral-Bold.ttf", 48)
            self.body_font = ImageFont.truetype("Inter-Regular.ttf", 32)
        except OSError:
            # Fallback to default
            self.title_font = ImageFont.load_default()
            self.body_font = ImageFont.load_default()
            
    def _create_base(self) -> Image.Image:
        """Create base card with background."""
        # Create dark base
        base = Image.new("RGBA", (self.width, self.height), COLORS["ink"])
        
        # Add frosted glass panel
        glass = Image.new("RGBA", (self.width-80, self.height-80), COLORS["ice"])
        glass.putalpha(100)  # Semi-transparent
        
        # Composite with offset for shadow
        base.paste(glass, (40, 40), glass)
        
        return base
        
    def _draw_flap_arrows(self, draw: ImageDraw.ImageDraw, flaps: List[str]):
        """Draw vector arrows for flap options."""
        for i, flap in enumerate(flaps):
            # Calculate arrow position
            x = self.width // 2 + 100
            y = 400 + i * 100
            
            # Draw arrow using matplotlib
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            if "rotation" in flap.lower():
                # Draw curved arrow
                arrow = FancyArrowPatch(
                    (0.2, 0.2), (0.8, 0.8),
                    connectionstyle="arc3,rad=0.3",
                    color=COLORS["gold"],
                    arrowstyle="->",
                    linewidth=2
                )
            else:
                # Draw straight arrow
                arrow = FancyArrowPatch(
                    (0.2, 0.5), (0.8, 0.5),
                    color=COLORS["gold"],
                    arrowstyle="->",
                    linewidth=2
                )
                
            ax.add_patch(arrow)
            ax.axis("off")
            
            # Convert to PIL and paste
            fig.canvas.draw()
            arrow_img = Image.frombytes(
                "RGBA",
                fig.canvas.get_width_height(),
                fig.canvas.tostring_argb(),
                "raw",
                "ARGB",
                0,
                1
            )
            
            plt.close(fig)
            
            # Add label
            draw.text((x+100, y), flap, font=self.body_font, fill=COLORS["ink"])
            
    def render(
        self,
        diagnosis: str,
        subunit: str,
        margins: Dict,
        gate_decision: Dict,
        flaps: List[str],
        danger_notes: List[Dict],
        citations: List[Dict],
        output_path: Path
    ) -> None:
        """Render a complete guideline card.
        
        Args:
            diagnosis: Diagnosis name 
            subunit: Facial subunit
            margins: Margin recommendations
            gate_decision: Gate decision with reason
            flaps: List of flap options
            danger_notes: List of danger structure notes
            citations: List of citation dicts
            output_path: Where to save the PNG
        """
        # Create base image
        img = self._create_base()
        draw = ImageDraw.Draw(img)
        
        # Title
        title = f"{diagnosis.title()} — {subunit.replace('_', ' ').title()}"
        draw.text((60, 60), title, font=self.title_font, fill=COLORS["gold"])
        
        # Left column
        y = 180
        
        # Margins
        draw.text(
            (60, y),
            f"Margins: {margins['peripheral']}",
            font=self.body_font,
            fill=COLORS["ink"]
        )
        y += 50
        draw.text(
            (60, y),
            f"Deep: {margins['deep']}",
            font=self.body_font,
            fill=COLORS["ink"]
        )
        y += 80
        
        # Gate decision
        color = COLORS["safe"] if gate_decision["allow"] else COLORS["alert"]
        draw.rectangle((60, y, 400, y+40), fill=color)
        draw.text(
            (70, y+5),
            "PROCEED" if gate_decision["allow"] else "DEFER",
            font=self.body_font,
            fill="white"
        )
        y += 60
        draw.text(
            (60, y),
            gate_decision["reason"],
            font=self.body_font,
            fill=COLORS["ink"]
        )
        
        # Right column - Flaps
        self._draw_flap_arrows(draw, flaps)
        
        # Danger notes
        y = self.height - 300
        for note in danger_notes:
            draw.text(
                (60, y),
                f"⚠ {note['notes']}",
                font=self.body_font,
                fill=COLORS["alert"]
            )
            y += 40
            
        # Citations
        y = self.height - 100
        citation_text = " • ".join(
            f"[{i+1}] {cit['title']}"
            for i, cit in enumerate(citations)
        )
        draw.text(
            (60, y),
            citation_text,
            font=self.body_font,
            fill=COLORS["gold"]
        )
        
        # Decision support banner
        draw.rectangle(
            (0, self.height-40, self.width, self.height),
            fill=COLORS["alert"]
        )
        draw.text(
            (self.width//2-100, self.height-35),
            "DECISION SUPPORT ONLY — NOT FOR CLINICAL USE",
            font=self.body_font,
            fill="white"
        )
        
        # Save
        img.save(output_path)
