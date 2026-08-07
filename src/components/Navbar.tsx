import React from 'react';

export type AppTab = 'comparison' | 'tactical' | 'history';

interface NavbarProps {
  activeTab: AppTab;
  onChangeTab: (tab: AppTab) => void;
  historyCount: number;
}

const tabs: Array<{ id: AppTab; label: string }> = [
  { id: 'comparison', label: 'Comparaison' },
  { id: 'tactical', label: 'Carte tactique' },
  { id: 'history', label: 'Historique' },
];

export const Navbar: React.FC<NavbarProps> = ({ activeTab, onChangeTab, historyCount }) => (
  <header className="app-header">
    <div className="brand-block">
      <span className="brand-mark" aria-hidden="true">BP</span>
      <div>
        <strong>Baysian Patrol</strong>
        <span>Monte-Carlo maritime</span>
      </div>
    </div>

    <nav className="primary-nav" aria-label="Navigation principale">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          className={activeTab === tab.id ? 'active' : ''}
          onClick={() => onChangeTab(tab.id)}
          aria-current={activeTab === tab.id ? 'page' : undefined}
        >
          {tab.label}
          {tab.id === 'history' && historyCount > 0 && <span className="history-count">{historyCount}</span>}
        </button>
      ))}
    </nav>

    <div className="version-block"><span>VERSION</span><strong>2.4.0</strong></div>
  </header>
);
