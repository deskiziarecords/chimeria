# terminal.py - Complete SMK Trading Terminal with Auto-Profit & Visual Signals

import os
import sys
import time
import threading
import queue
from datetime import datetime
from typing import Dict, Optional
import curses
from dataclasses import dataclass

# Import your existing modules
from smk_fyers_integration import SMKFyersManager
from enhanced_order_management import EnhancedOrderManagementSystem


class SMKTradingTerminal:
    """
    Complete Trading Terminal with:
    - Real-time λ sensor display (λ₁-λ₈)
    - Profit/position tracking
    - Auto-profit target status (2% adjustable)
    - Order management interface
    - Visual signals (colors, flashing)
    - Key bindings for trading actions
    """
    
    def __init__(self, fyers_manager: SMKFyersManager, oms: EnhancedOrderManagementSystem):
        self.fyers = fyers_manager
        self.oms = oms
        self.running = True
        self.stdscr = None
        
        # Data queues
        self.signal_queue = queue.Queue()
        self.profit_queue = queue.Queue()
        self.order_queue = queue.Queue()
        
        # Register callbacks
        self.fyers.register_signal_callback(self._on_signal)
        
        # Cursor position
        self.selected_row = 0
        self.menu_items = [
            "Dashboard",
            "Positions",
            "Active Orders",
            "Profit Targets",
            "Trade History",
            "Settings",
            "Exit"
        ]
        
    def _on_signal(self, signal: Dict):
        """Handle incoming signals from SMK"""
        self.signal_queue.put(signal)
    
    def _profit_callback(self, profit_data: Dict):
        """Handle profit updates"""
        self.profit_queue.put(profit_data)
    
    def _order_callback(self, order_data: Dict):
        """Handle order updates"""
        self.order_queue.put(order_data)
    
    def init_curses(self):
        """Initialize curses for terminal UI"""
        self.stdscr = curses.initscr()
        curses.start_color()
        curses.use_default_colors()
        curses.curs_set(0)  # Hide cursor
        self.stdscr.nodelay(True)  # Non-blocking input
        
        # Initialize color pairs
        curses.init_pair(1, curses.COLOR_GREEN, -1)      # Positive/Profit
        curses.init_pair(2, curses.COLOR_RED, -1)        # Negative/Loss
        curses.init_pair(3, curses.COLOR_YELLOW, -1)     # Warning/Signal
        curses.init_pair(4, curses.COLOR_CYAN, -1)       # Info/Active
        curses.init_pair(5, curses.COLOR_MAGENTA, -1)    # λ sensor
        curses.init_pair(6, curses.COLOR_WHITE, -1)      # Normal
        curses.init_pair(7, curses.COLOR_BLUE, -1)       # Buy signal
        curses.init_pair(8, curses.COLOR_RED, -1)        # Sell signal
        curses.init_pair(9, curses.COLOR_GREEN, -1)      # Buy glow
        curses.init_pair(10, curses.COLOR_RED, -1)       # Sell glow
        
        # Get terminal dimensions
        self.height, self.width = self.stdscr.getmaxyx()
    
    def cleanup_curses(self):
        """Clean up curses"""
        curses.nocbreak()
        self.stdscr.keypad(False)
        curses.echo()
        curses.endwin()
    
    def draw_header(self):
        """Draw terminal header"""
        title = "╔══════════════════════════════════════════════════════════════════════════════╗"
        subtitle = "║                    SMK TRADING TERMINAL - AUTO-PROFIT (2%)                    ║"
        
        try:
            self.stdscr.attron(curses.A_BOLD | curses.color_pair(4))
            self.stdscr.addstr(0, 0, title[:self.width-1])
            self.stdscr.addstr(1, 0, subtitle[:self.width-1])
            self.stdscr.attroff(curses.A_BOLD)
        except:
            pass
    
    def draw_lambda_panel(self, start_y: int, lambda_scores: Dict):
        """Draw λ sensor panel (λ₁-λ₈)"""
        self.stdscr.attron(curses.A_BOLD)
        self.stdscr.addstr(start_y, 2, "╔══════════════════════════════════════════════════════════╗")
        self.stdscr.addstr(start_y+1, 2, "║                    λ SENSOR DASHBOARD                    ║")
        self.stdscr.addstr(start_y+2, 2, "╠══════════════════════════════════════════════════════════╣")
        self.stdscr.attroff(curses.A_BOLD)
        
        sensors = [
            ("λ₁", "ENTRAPMENT", lambda_scores.get('λ₁_entrapment', 0), 0.7, "🔴", "🟢"),
            ("λ₃", "HARMONIC", lambda_scores.get('λ₃_harmonic', 0), 0.5, "🔴", "🟢"),
            ("λ₄", "MANIPULATION", lambda_scores.get('λ₄_manipulation', 0), 0.5, "⚡", "✓"),
            ("λ₆", "DISPLACEMENT", lambda_scores.get('λ₆_displacement', 0), 0.6, "💨", "📈"),
            ("λ₇", "MACRO GATE", lambda_scores.get('λ₇_macro', 0.5), 0.3, "🌍", "✅"),
            ("λ₈", "LIGHT-CONE", lambda_scores.get('λ₈_light_cone', 0), 0.7, "⚠️", "🔒"),
        ]
        
        row = start_y + 3
        for i, (name, desc, value, threshold, warn_icon, ok_icon) in enumerate(sensors):
            # Bar visualization
            bar_len = int(value * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            
            # Color based on value
            if value >= threshold:
                color = curses.color_pair(1) if name != "λ₈" else curses.color_pair(2)
                status_icon = ok_icon
            else:
                color = curses.color_pair(3)
                status_icon = warn_icon
            
            self.stdscr.attron(color)
            self.stdscr.addstr(row, 4, f"{name} {desc:12} [{bar}] {value*100:5.1f}% {status_icon}")
            self.stdscr.attroff(color)
            row += 1
        
        # Fusion score
        fusion = lambda_scores.get('fusion_confidence', 0)
        fusion_color = curses.color_pair(1) if fusion > 0.6 else curses.color_pair(3)
        self.stdscr.attron(curses.A_BOLD | fusion_color)
        self.stdscr.addstr(row+1, 4, f"FUSION SCORE: {fusion*100:.1f}%")
        self.stdscr.attroff(curses.A_BOLD | fusion_color)
    
    def draw_positions_panel(self, start_y: int, positions: Dict):
        """Draw active positions panel"""
        self.stdscr.attron(curses.A_BOLD)
        self.stdscr.addstr(start_y, 42, "╔══════════════════════════════════════════════════════════╗")
        self.stdscr.addstr(start_y+1, 42, "║                   ACTIVE POSITIONS                      ║")
        self.stdscr.addstr(start_y+2, 42, "╠══════════════════════════════════════════════════════════╣")
        self.stdscr.attroff(curses.A_BOLD)
        
        if not positions:
            self.stdscr.addstr(start_y+3, 44, "No active positions")
            return
        
        row = start_y + 3
        self.stdscr.addstr(row, 44, f"{'ID':<8} {'Symbol':<12} {'Side':<6} {'Qty':<8} {'Entry':<10} {'Current':<10} {'P&L':<10} {'Target':<8}")
        row += 1
        
        for pos_id, pos in list(positions.items())[:8]:  # Show max 8 positions
            profit_pct = pos.current_profit_pct
            target_pct = pos.profit_target.value
            
            # Color based on profit
            if profit_pct >= target_pct:
                color = curses.color_pair(2)  # Target reached
                status = "🏆"
            elif profit_pct > 0:
                color = curses.color_pair(1)
                status = "📈"
            else:
                color = curses.color_pair(2)
                status = "📉"
            
            self.stdscr.attron(color)
            self.stdscr.addstr(row, 44, f"{pos_id[:8]:<8} {pos.symbol[:12]:<12} {pos.side:<6} {pos.quantity:<8} {pos.entry_price:<10.2f} {pos.current_price:<10.2f} {profit_pct:>+6.2f}% {status}")
            self.stdscr.attroff(color)
            
            # Progress bar for profit target
            progress = min(100, int((profit_pct / target_pct) * 40))
            if progress > 0:
                bar = "█" * progress + "░" * (40 - progress)
                bar_color = curses.color_pair(1) if profit_pct > 0 else curses.color_pair(2)
                self.stdscr.attron(bar_color)
                self.stdscr.addstr(row+1, 44, f"      Target: {bar}")
                self.stdscr.attroff(bar_color)
            
            row += 2
    
    def draw_profit_panel(self, start_y: int, profit_target: float):
        """Draw profit target panel"""
        self.stdscr.attron(curses.A_BOLD)
        self.stdscr.addstr(start_y, 2, "╔══════════════════════════════════════════════════════════╗")
        self.stdscr.addstr(start_y+1, 2, "║                    PROFIT TARGET                        ║")
        self.stdscr.addstr(start_y+2, 2, "╚══════════════════════════════════════════════════════════╝")
        self.stdscr.attroff(curses.A_BOLD)
        
        self.stdscr.addstr(start_y+3, 4, f" Current Target: {profit_target}%")
        self.stdscr.addstr(start_y+4, 4, " Press 't' to change target (1-5%)")
        self.stdscr.addstr(start_y+5, 4, " Auto-close enabled: YES")
    
    def draw_status_panel(self, start_y: int, status: Dict):
        """Draw status panel"""
        self.stdscr.attron(curses.A_BOLD)
        self.stdscr.addstr(start_y, 42, "╔══════════════════════════════════════════════════════════╗")
        self.stdscr.addstr(start_y+1, 42, "║                     SYSTEM STATUS                       ║")
        self.stdscr.addstr(start_y+2, 42, "╚══════════════════════════════════════════════════════════╝")
        self.stdscr.attroff(curses.A_BOLD)
        
        auth_status = " AUTHENTICATED" if self.fyers.is_authenticated() else "❌ NOT AUTHENTICATED"
        auth_color = curses.color_pair(1) if self.fyers.is_authenticated() else curses.color_pair(2)
        
        self.stdscr.attron(auth_color)
        self.stdscr.addstr(start_y+3, 44, auth_status)
        self.stdscr.attroff(auth_color)
        
        self.stdscr.addstr(start_y+4, 44, f" Active Orders: {status.get('active_orders', 0)}")
        self.stdscr.addstr(start_y+5, 44, f" Total P&L: {status.get('total_profit', 0):+.2f}%")
        self.stdscr.addstr(start_y+6, 44, f" Win Rate: {status.get('win_rate', 0):.1f}%")
        
        # Live signal indicator
        if status.get('last_signal'):
            signal = status['last_signal']
            if signal.get('side') == 'BUY':
                self.stdscr.attron(curses.A_BOLD | curses.color_pair(9))
                self.stdscr.addstr(start_y+8, 44, "🔴🔴🔴 BUY SIGNAL ACTIVE 🔴🔴🔴")
                self.stdscr.attroff(curses.A_BOLD | curses.color_pair(9))
            elif signal.get('side') == 'SELL':
                self.stdscr.attron(curses.A_BOLD | curses.color_pair(10))
                self.stdscr.addstr(start_y+8, 44, "🔵🔵🔵 SELL SIGNAL ACTIVE 🔵🔵🔵")
                self.stdscr.attroff(curses.A_BOLD | curses.color_pair(10))
    
    def draw_menu(self):
        """Draw navigation menu"""
        menu_y = self.height - 3
        
        self.stdscr.attron(curses.A_BOLD)
        self.stdscr.addstr(menu_y - 2, 2, "╔══════════════════════════════════════════════════════════════════════════════╗")
        self.stdscr.addstr(menu_y - 1, 2, "║                                    MENU                                      ║")
        self.stdscr.addstr(menu_y, 2, "╚══════════════════════════════════════════════════════════════════════════════╝")
        self.stdscr.attroff(curses.A_BOLD)
        
        # Draw menu items
        x = 4
        for i, item in enumerate(self.menu_items):
            if i == self.selected_row:
                self.stdscr.attron(curses.A_REVERSE | curses.A_BOLD)
                self.stdscr.addstr(menu_y + 1, x, item)
                self.stdscr.attroff(curses.A_REVERSE | curses.A_BOLD)
            else:
                self.stdscr.addstr(menu_y + 1, x, item)
            x += len(item) + 4
    
    def draw_notifications(self):
        """Draw notification area"""
        notif_y = self.height - 8
        
        # Process signal queue
        signals = []
        while not self.signal_queue.empty():
            signals.append(self.signal_queue.get())
        
        for i, signal in enumerate(signals[-3:]):  # Show last 3 signals
            if signal.get('type') == 'TRADE_EXECUTED':
                color = curses.color_pair(1) if signal.get('side') == 'BUY' else curses.color_pair(2)
                self.stdscr.attron(curses.A_BOLD | color)
                msg = f" {signal.get('side')} ORDER: {signal.get('quantity')} @ {signal.get('price'):.2f} | Target: {signal.get('profit_target')}%"
                self.stdscr.addstr(notif_y + i, 4, msg)
                self.stdscr.attroff(curses.A_BOLD | color)
            
            elif signal.get('type') == 'PROFIT_ACHIEVED':
                self.stdscr.attron(curses.A_BOLD | curses.color_pair(3))
                msg = f" PROFIT TARGET HIT: {signal.get('profit_pct'):.2f}% - {signal.get('reason')}"
                self.stdscr.addstr(notif_y + i, 4, msg)
                self.stdscr.attroff(curses.A_BOLD | curses.color_pair(3))
            
            elif signal.get('type') == 'VETO':
                self.stdscr.attron(curses.A_BOLD | curses.color_pair(2))
                msg = f" VETO: {signal.get('reason')}"
                self.stdscr.addstr(notif_y + i, 4, msg)
                self.stdscr.attroff(curses.A_BOLD | curses.color_pair(2))
    
    def handle_input(self):
        """Handle keyboard input"""
        try:
            key = self.stdscr.getch()
            
            if key == ord('q') or key == ord('Q'):
                self.running = False
            
            elif key == curses.KEY_UP:
                self.selected_row = (self.selected_row - 1) % len(self.menu_items)
            
            elif key == curses.KEY_DOWN:
                self.selected_row = (self.selected_row + 1) % len(self.menu_items)
            
            elif key == ord('\n') or key == ord(' '):  # Enter or Space
                self.execute_menu_action()
            
            elif key == ord('t') or key == ord('T'):
                self.change_profit_target()
            
            elif key == ord('c') or key == ord('C'):
                self.close_all_positions()
            
        except:
            pass
    
    def execute_menu_action(self):
        """Execute selected menu action"""
        action = self.menu_items[self.selected_row]
        
        if action == " Dashboard":
            pass  # Already on dashboard
        elif action == " Positions":
            self.show_positions_detail()
        elif action == " Active Orders":
            self.show_orders_detail()
        elif action == " Profit Targets":
            self.change_profit_target()
        elif action == " Trade History":
            self.show_trade_history()
        elif action == " Settings":
            self.show_settings()
        elif action == " Exit":
            self.running = False
    
    def change_profit_target(self):
        """Interactive profit target change"""
        curses.echo()
        self.stdscr.nodelay(False)
        
        self.stdscr.attron(curses.A_BOLD | curses.color_pair(3))
        self.stdscr.addstr(self.height - 5, 4, "Enter new profit target (1-5%): ")
        self.stdscr.attroff(curses.A_BOLD | curses.color_pair(3))
        
        try:
            target_input = self.stdscr.getstr().decode('utf-8')
            new_target = float(target_input)
            if 1.0 <= new_target <= 5.0:
                self.fyers.set_profit_target(new_target)
                self.stdscr.addstr(self.height - 4, 4, f"✅ Profit target updated to {new_target}%")
            else:
                self.stdscr.addstr(self.height - 4, 4, "❌ Target must be between 1% and 5%")
        except:
            self.stdscr.addstr(self.height - 4, 4, "❌ Invalid input")
        
        self.stdscr.addstr(self.height - 3, 4, "Press any key to continue...")
        self.stdscr.getch()
        
        curses.noecho()
        self.stdscr.nodelay(True)
    
    def close_all_positions(self):
        """Close all open positions"""
        self.stdscr.attron(curses.A_BOLD | curses.color_pair(2))
        self.stdscr.addstr(self.height - 5, 4, "⚠️ CLOSE ALL POSITIONS? (y/N): ")
        self.stdscr.attroff(curses.A_BOLD | curses.color_pair(2))
        
        self.stdscr.nodelay(False)
        confirm = self.stdscr.getch()
        self.stdscr.nodelay(True)
        
        if confirm == ord('y') or confirm == ord('Y'):
            for pos_id in list(self.fyers.active_positions.keys()):
                self.fyers.close_position(pos_id, "Manual close")
            self.stdscr.addstr(self.height - 4, 4, "✅ All positions closed")
    
    def show_positions_detail(self):
        """Show detailed positions view"""
        self.stdscr.clear()
        self.stdscr.addstr(0, 0, "=== ACTIVE POSITIONS DETAIL ===")
        row = 2
        
        for pos_id, pos in self.fyers.active_positions.items():
            self.stdscr.addstr(row, 2, f"ID: {pos_id}")
            self.stdscr.addstr(row+1, 4, f"Symbol: {pos.symbol}")
            self.stdscr.addstr(row+2, 4, f"Side: {pos.side}")
            self.stdscr.addstr(row+3, 4, f"Quantity: {pos.quantity}")
            self.stdscr.addstr(row+4, 4, f"Entry: {pos.entry_price:.2f}")
            self.stdscr.addstr(row+5, 4, f"Current: {pos.current_price:.2f}")
            self.stdscr.addstr(row+6, 4, f"P&L: {pos.current_profit_pct:+.2f}%")
            self.stdscr.addstr(row+7, 4, f"Target: {pos.profit_target.value}%")
            row += 10
        
        self.stdscr.addstr(row+2, 2, "Press any key to return...")
        self.stdscr.getch()
    
    def show_orders_detail(self):
        """Show orders detail"""
        self.stdscr.clear()
        self.stdscr.addstr(0, 0, "=== ACTIVE ORDERS ===")
        
        orders = self.fyers.get_order_book()
        row = 2
        
        for order in orders[:20]:
            self.stdscr.addstr(row, 2, f"ID: {order.get('id', 'N/A')}")
            self.stdscr.addstr(row+1, 4, f"Symbol: {order.get('symbol', 'N/A')}")
            self.stdscr.addstr(row+2, 4, f"Status: {order.get('status', 'N/A')}")
            row += 5
        
        self.stdscr.addstr(row+2, 2, "Press any key to return...")
        self.stdscr.getch()
    
    def show_trade_history(self):
        """Show trade history"""
        self.stdscr.clear()
        self.stdscr.addstr(0, 0, "=== TRADE HISTORY ===")
        row = 2
        
        stats = self.fyers.get_statistics()
        self.stdscr.addstr(row, 2, f"Total Trades: {stats.get('total_trades', 0)}")
        self.stdscr.addstr(row+1, 2, f"Win Rate: {stats.get('win_rate', 0):.1f}%")
        self.stdscr.addstr(row+2, 2, f"Total Profit: {stats.get('total_profit', 0):+.2f}%")
        row += 5
        
        for pos in self.fyers.closed_positions[-20:]:
            self.stdscr.addstr(row, 2, f"{pos.symbol} | {pos.side} | {pos.quantity} | P&L: {pos.current_profit_pct:+.2f}%")
            row += 1
        
        self.stdscr.addstr(row+2, 2, "Press any key to return...")
        self.stdscr.getch()
    
    def show_settings(self):
        """Show settings panel"""
        self.stdscr.clear()
        self.stdscr.addstr(0, 0, "=== SETTINGS ===")
        self.stdscr.addstr(2, 2, "1. Change Profit Target (Current: {self.fyers.default_profit_target_pct}%)")
        self.stdscr.addstr(3, 2, "2. Toggle Auto-close")
        self.stdscr.addstr(4, 2, "3. Set Max Position Size")
        self.stdscr.addstr(5, 2, "4. Back")
        
        choice = self.stdscr.getch()
        if choice == ord('1'):
            self.change_profit_target()
    
    def run(self):
        """Main terminal loop"""
        self.init_curses()
        
        try:
            while self.running:
                self.stdscr.clear()
                
                # Get current data
                lambda_scores = {}
                if hasattr(self.fyers, 'lambda7') and self.fyers.lambda7:
                    # Get real λ scores from SMK
                    pass
                
                positions = self.fyers.active_positions
                stats = self.fyers.get_statistics()
                
                # Draw UI panels
                self.draw_header()
                self.draw_lambda_panel(3, lambda_scores)
                self.draw_positions_panel(3, positions)
                self.draw_profit_panel(17, self.fyers.default_profit_target_pct)
                self.draw_status_panel(17, stats)
                self.draw_notifications()
                self.draw_menu()
                
                # Refresh display
                self.stdscr.refresh()
                
                # Handle input
                self.handle_input()
                
                # Update rate
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup_curses()


def main():
    """Main entry point"""
    print("Starting SMK Trading Terminal...")
    
    # Initialize components
    fyers_manager = SMKFyersManager()
    oms = EnhancedOrderManagementSystem(fyers_manager)
    
    # Start profit monitoring
    fyers_manager.start_profit_monitoring()
    
    # Start terminal
    terminal = SMKTradingTerminal(fyers_manager, oms)
    
    try:
        terminal.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        fyers_manager.stop_profit_monitoring()
        print("Terminal closed.")


if __name__ == "__main__":
    main()
