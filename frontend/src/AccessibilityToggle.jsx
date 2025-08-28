import React from 'react';

const AccessibilityToggle = ({ isEnabled, onToggle }) => {
  return (
    <div className="toggle-container">
      <span className="toggle-label">Accessibility Mode</span>
      <label className="switch">
        <input type="checkbox" checked={isEnabled} onChange={onToggle} />
        <span className="slider round"></span>
      </label>
    </div>
  );
};

export default AccessibilityToggle;