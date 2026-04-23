# order_management.py - Full order execution with λ sensor integration

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import uuid
import asyncio


class OrderType(Enum):
    """Order types supported by SMK"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class Order:
    """Single order with SMK metadata"""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    limit_price: Optional[float] = None
    trailing_percent: Optional[float] = None
    
    # SMK-specific fields
    lambda_scores: Dict[str, float] = field(default_factory=dict)  # λ₁-λ₈ scores
    risk_pct: float = 0.0
    confidence: float = 0.0
    veto_status: bool = False
    
    # Execution tracking
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Venue routing (for AEGIS)
    venue_allocation: Dict[str, float] = field(default_factory=dict)
    
    def is_active(self) -> bool:
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
    
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity


class OrderManager:
    """
    Advanced order management for SMK with:
    - Smart order routing (SchurRouter from AEGIS)
    - Veto integration (λ₇, λ₈)
    - Risk-based position sizing
    - Order splitting and aggregation
    """
    
    def __init__(self, 
                 broker_api,
                 schur_router=None,
                 aegis_bridge=None,
                 risk_manager=None):
        """
        Initialize Order Manager.
        
        Args:
            broker_api: Broker connection (Bitget, OANDA, etc.)
            schur_router: AEGIS venue router (61.8% dark / 38.2% lit)
            aegis_bridge: AEGIS execution bridge
            risk_manager: RiskManager instance for position checks
        """
        self.broker = broker_api
        self.schur_router = schur_router
        self.aegis_bridge = aegis_bridge
        self.risk_manager = risk_manager
        
        self.orders: Dict[str, Order] = {}
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        
        # Execution configuration
        self.max_slippage_bps = 10  # 10 basis points
        self.order_split_size = 100  # Split large orders into chunks
        
    def create_order(self,
                     symbol: str,
                     side: OrderSide,
                     quantity: float,
                     order_type: OrderType = OrderType.MARKET,
                     price: Optional[float] = None,
                     stop_price: Optional[float] = None,
                     limit_price: Optional[float] = None,
                     lambda_scores: Optional[Dict[str, float]] = None,
                     confidence: float = 0.0) -> Optional[Order]:
        """
        Create and validate an order with λ sensor checks.
        
        Returns:
            Order object if validated, None if vetoed by λ sensors
        """
        # 1. Risk check from RiskManager
        if self.risk_manager:
            position_size_ok = self.risk_manager.calculate_position_size(
                price or 0, 
                risk_per_trade=0.02
            )
            if position_size_ok <= 0:
                print(f"❌ Order rejected: Position size zero based on risk limits")
                return None
        
        # 2. Create order
        order_id = str(uuid.uuid4())[:8]
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            limit_price=limit_price,
            lambda_scores=lambda_scores or {},
            confidence=confidence,
            status=OrderStatus.PENDING
        )
        
        # 3. Apply AEGIS SchurRouter for venue allocation
        if self.schur_router:
            order.venue_allocation = self.schur_router.route(quantity)
            print(f"📍 Venue allocation: {order.venue_allocation}")
        
        # 4. Check λ₇ macro veto
        if lambda_scores and lambda_scores.get('λ₇_macro', 0) < 0.3:
            order.veto_status = True
            order.status = OrderStatus.REJECTED
            print(f"🚫 Order vetoed by λ₇ Macro Gate (score: {lambda_scores.get('λ₇_macro', 0):.2f})")
            return None
        
        # 5. Check λ₈ light-cone violation
        if lambda_scores and lambda_scores.get('λ₈_light_cone', 0) > 0.7:
            order.veto_status = True
            order.status = OrderStatus.REJECTED
            print(f"🚫 Order vetoed by λ₈ Light-Cone Violation (score: {lambda_scores.get('λ₈_light_cone', 0):.2f})")
            return None
        
        # Store order
        self.orders[order_id] = order
        self.active_orders[order_id] = order
        
        print(f"✅ Order created: {order.side.value} {order.quantity} {order.symbol} "
              f"(Confidence: {order.confidence:.2%})")
        
        return order
    
    async def execute_order(self, order: Order) -> Dict:
        """
        Execute order with smart routing and slippage control.
        
        Returns execution result with fill details.
        """
        if not order.is_active():
            return {'success': False, 'reason': f'Order not active: {order.status.value}'}
        
        if order.veto_status:
            return {'success': False, 'reason': 'Order vetoed by λ sensors'}
        
        # Split large orders to reduce market impact
        order_chunks = self._split_order(order)
        
        executions = []
        total_filled = 0
        total_cost = 0
        
        for chunk_qty in order_chunks:
            try:
                # Execute chunk based on order type
                if order.order_type == OrderType.MARKET:
                    result = await self._execute_market_chunk(order, chunk_qty)
                elif order.order_type == OrderType.LIMIT:
                    result = await self._execute_limit_chunk(order, chunk_qty)
                elif order.order_type == OrderType.STOP:
                    result = await self._execute_stop_chunk(order, chunk_qty)
                else:
                    result = await self._execute_market_chunk(order, chunk_qty)
                
                if result['success']:
                    executions.append(result)
                    total_filled += result['filled']
                    total_cost += result['cost']
                    
                    # Apply venue routing if configured
                    if order.venue_allocation and self.schur_router:
                        await self._route_to_venues(order, chunk_qty, result['price'])
                        
            except Exception as e:
                print(f"❌ Order chunk failed: {e}")
                break
        
        # Update order status
        if total_filled >= order.quantity:
            order.status = OrderStatus.FILLED
            order.filled_quantity = total_filled
            order.average_fill_price = total_cost / total_filled if total_filled > 0 else 0
        elif total_filled > 0:
            order.status = OrderStatus.PARTIALLY_FILLED
            order.filled_quantity = total_filled
            order.average_fill_price = total_cost / total_filled if total_filled > 0 else 0
        
        order.updated_at = datetime.now()
        
        # Update RiskManager position
        if self.risk_manager and total_filled > 0:
            self.risk_manager.add_position(
                symbol=order.symbol,
                quantity=total_filled if order.side == OrderSide.BUY else -total_filled,
                price=order.average_fill_price
            )
        
        return {
            'success': total_filled > 0,
            'order_id': order.id,
            'filled_quantity': total_filled,
            'average_price': order.average_fill_price,
            'executions': executions,
            'venue_allocation': order.venue_allocation
        }
    
    async def _execute_market_chunk(self, order: Order, quantity: float) -> Dict:
        """Execute market order chunk"""
        try:
            # Get current market price
            current_price = await self.broker.get_price(order.symbol)
            
            # Check slippage
            if order.price:
                slippage = abs(current_price - order.price) / order.price
                if slippage > (self.max_slippage_bps / 10000):
                    print(f"⚠️ Slippage too high: {slippage*100:.2f}% > {self.max_slippage_bps/100:.2f}%")
                    return {'success': False, 'reason': 'Excessive slippage'}
            
            # Execute
            fill = await self.broker.market_order(
                symbol=order.symbol,
                side=order.side.value,
                quantity=quantity
            )
            
            return {
                'success': True,
                'filled': fill['filled'],
                'price': fill['price'],
                'cost': fill['filled'] * fill['price']
            }
            
        except Exception as e:
            return {'success': False, 'reason': str(e)}
    
    async def _execute_limit_chunk(self, order: Order, quantity: float) -> Dict:
        """Execute limit order chunk"""
        if not order.limit_price:
            return {'success': False, 'reason': 'No limit price'}
        
        try:
            fill = await self.broker.limit_order(
                symbol=order.symbol,
                side=order.side.value,
                quantity=quantity,
                limit_price=order.limit_price
            )
            
            return {
                'success': True,
                'filled': fill['filled'],
                'price': fill['price'],
                'cost': fill['filled'] * fill['price']
            }
            
        except Exception as e:
            return {'success': False, 'reason': str(e)}
    
    async def _execute_stop_chunk(self, order: Order, quantity: float) -> Dict:
        """Execute stop order chunk"""
        if not order.stop_price:
            return {'success': False, 'reason': 'No stop price'}
        
        try:
            fill = await self.broker.stop_order(
                symbol=order.symbol,
                side=order.side.value,
                quantity=quantity,
                stop_price=order.stop_price
            )
            
            return {
                'success': True,
                'filled': fill['filled'],
                'price': fill['price'],
                'cost': fill['filled'] * fill['price']
            }
            
        except Exception as e:
            return {'success': False, 'reason': str(e)}
    
    async def _route_to_venues(self, order: Order, quantity: float, price: float):
        """Route order to multiple venues using SchurRouter"""
        if not order.venue_allocation:
            return
        
        for venue, fraction in order.venue_allocation.items():
            venue_qty = quantity * fraction
            if venue_qty > 0:
                try:
                    await self.broker.venue_order(
                        venue=venue,
                        symbol=order.symbol,
                        side=order.side.value,
                        quantity=venue_qty,
                        price=price
                    )
                    print(f"  📍 Routed {venue_qty:.2f} to {venue}")
                except Exception as e:
                    print(f"  ❌ Venue {venue} failed: {e}")
    
    def _split_order(self, order: Order) -> List[float]:
        """Split large orders into smaller chunks"""
        if order.quantity <= self.order_split_size:
            return [order.quantity]
        
        num_chunks = int(order.quantity / self.order_split_size)
        chunk_size = order.quantity / num_chunks
        
        return [chunk_size] * num_chunks
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order"""
        if order_id in self.active_orders:
            order = self.active_orders[order_id]
            order.status = OrderStatus.CANCELLED
            del self.active_orders[order_id]
            print(f"🛑 Order {order_id} cancelled")
            return True
        return False
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status"""
        return self.orders.get(order_id)
    
    def get_active_orders(self) -> List[Order]:
        """Get all active orders"""
        return list(self.active_orders.values())
    
    def get_order_summary(self) -> Dict:
        """Get order statistics"""
        return {
            'total_orders': len(self.orders),
            'active_orders': len(self.active_orders),
            'filled_orders': sum(1 for o in self.orders.values() if o.status == OrderStatus.FILLED),
            'rejected_orders': sum(1 for o in self.orders.values() if o.status == OrderStatus.REJECTED),
            'total_volume': sum(o.quantity for o in self.orders.values() if o.status == OrderStatus.FILLED),
            'avg_confidence': sum(o.confidence for o in self.orders.values()) / len(self.orders) if self.orders else 0
        }


# ---------------------------------------------------------------------------
# Integration with SMK Pipeline
# ---------------------------------------------------------------------------

class SMKLiveExecutionManager:
    """
    Complete execution manager connecting SMK pipeline to order management.
    
    This integrates:
    - λ₇ (Macro Gate) for order validation
    - λ₈ (Light-Cone) for kill switch
    - AEGIS SchurRouter for venue allocation
    - RiskManager for position sizing
    """
    
    def __init__(self, 
                 broker_api,
                 initial_capital: float = 100_000):
        
        self.order_manager = OrderManager(
            broker_api=broker_api,
            schur_router=self._create_schur_router(),
            risk_manager=None  # Will be set later
        )
        
        self.risk_manager = None  # To be initialized
        self.active_positions = {}
        self.trade_log = []
        
    def _create_schur_router(self):
        """Create AEGIS SchurRouter for venue allocation"""
        # 61.8% dark pools, 38.2% lit exchanges (Fibonacci ratio)
        class SchurRouter:
            def __init__(self):
                self.venues = {
                    'DARK_POOL_1': 0.382,  # 38.2%
                    'DARK_POOL_2': 0.236,  # 23.6%
                    'LIT_EXCHANGE_1': 0.236,  # 23.6%
                    'LIT_EXCHANGE_2': 0.146   # 14.6%
                }
            
            def route(self, quantity):
                return {venue: quantity * frac for venue, frac in self.venues.items()}
        
        return SchurRouter()
    
    async def on_smk_signal(self, 
                           smk_result: Dict,
                           lambda7_score: float,
                           lambda8_score: float,
                           fusion_p: float,
                           current_price: float) -> Optional[Order]:
        """
        Called when SMK generates a trading signal.
        
        Args:
            smk_result: Full SMKPipeline.step() result
            lambda7_score: λ₇ macro gate score (0-1)
            lambda8_score: λ₈ light-cone score (0-1)
            fusion_p: Fused signal from lambda_fusion_engine
            current_price: Current market price
        """
        
        # 1. Determine trade direction
        if fusion_p > 0.3:
            side = OrderSide.BUY
            confidence = fusion_p
        elif fusion_p < -0.3:
            side = OrderSide.SELL
            confidence = abs(fusion_p)
        else:
            print("⏸️ No signal - holding")
            return None
        
        # 2. Calculate position size from RiskManager
        vol_decay = smk_result.get('vol_decay', {})
        position_size = self._calculate_position_size(
            current_price=current_price,
            confidence=confidence,
            entrapment=vol_decay.get('entrapped', False),
            manipulation=smk_result.get('manipulation', {}).get('active', False),
            lambda7_score=lambda7_score,
            lambda8_score=lambda8_score
        )
        
        if position_size <= 0:
            print("❌ Position size zero - no trade")
            return None
        
        # 3. Create order with λ sensor metadata
        order = self.order_manager.create_order(
            symbol='EURUSD',  # or from smk_result
            side=side,
            quantity=position_size,
            order_type=OrderType.MARKET,  # or LIMIT based on expansion
            price=current_price,
            lambda_scores={
                'λ₁_entrapment': 1.0 if vol_decay.get('entrapped') else 0.0,
                'λ₄_manipulation': 1.0 if smk_result.get('manipulation', {}).get('active') else 0.0,
                'λ₆_displacement': smk_result.get('displacement', {}).get('body_ratio', 0),
                'λ₇_macro': lambda7_score,
                'λ₈_light_cone': lambda8_score
            },
            confidence=confidence
        )
        
        # 4. Execute if validated
        if order and not order.veto_status:
            result = await self.order_manager.execute_order(order)
            
            if result['success']:
                self.trade_log.append({
                    'timestamp': datetime.now(),
                    'order_id': order.id,
                    'side': side.value,
                    'quantity': result['filled_quantity'],
                    'price': result['average_price'],
                    'lambda7': lambda7_score,
                    'lambda8': lambda8_score,
                    'confidence': confidence
                })
                
                print(f"\n🎯 TRADE EXECUTED:")
                print(f"   Side: {side.value}")
                print(f"   Size: {result['filled_quantity']:.2f}")
                print(f"   Price: {result['average_price']:.5f}")
                print(f"   Confidence: {confidence:.2%}")
                print(f"   λ₇: {lambda7_score:.2f} | λ₈: {lambda8_score:.2f}")
                
                return order
        
        return None
    
    def _calculate_position_size(self,
                                 current_price: float,
                                 confidence: float,
                                 entrapment: bool,
                                 manipulation: bool,
                                 lambda7_score: float,
                                 lambda8_score: float) -> float:
        """
        Calculate position size based on λ sensor scores and risk.
        
        Base size = $10,000 / price (for 0.1 BTC or 0.1 lots)
        Then adjusted by:
        - Confidence (0.5-1.5x)
        - Entrapment (1.2x boost)
        - Manipulation (0.6x reduction)
        - λ₇ macro (0-1.2x)
        - λ₈ light-cone (0-1x)
        """
        base_size = 10000 / current_price  # $10k notional
        
        # Multipliers
        confidence_mult = 0.5 + confidence  # 0.5 to 1.5
        entrapment_mult = 1.2 if entrapment else 1.0
        manipulation_mult = 0.6 if manipulation else 1.0
        macro_mult = lambda7_score  # 0 to 1
        light_cone_mult = 1 - lambda8_score  # Inverse: higher λ₈ = smaller size
        
        final_size = (base_size * 
                     confidence_mult * 
                     entrapment_mult * 
                     manipulation_mult * 
                     macro_mult * 
                     light_cone_mult)
        
        return round(final_size, 2)
    
    def get_summary(self) -> Dict:
        """Get execution summary"""
        return {
            'total_trades': len(self.trade_log),
            'active_orders': len(self.order_manager.get_active_orders()),
            'recent_trades': self.trade_log[-5:],
            'order_stats': self.order_manager.get_order_summary()
        }


# ---------------------------------------------------------------------------
# Example: Complete Pipeline
# ---------------------------------------------------------------------------

async def main():
    """Example of full SMK -> Order execution pipeline"""
    
    # 1. Initialize
    from your_smk_pipeline import SMKPipeline
    from lambda_fusion_engine import LambdaFusionEngine
    from macro_causality_gate import Lambda7MacroGate
    from light_cone_violation import LightConeViolationDetector
    
    # 2. Setup components
    pipeline = SMKPipeline()
    fusion = LambdaFusionEngine()
    lambda7 = Lambda7MacroGate()
    lambda8 = LightConeViolationDetector()
    
    # 3. Setup execution (with your broker)
    # from your_broker_connector import BitgetBroker
    # broker = BitgetBroker(api_key='...', api_secret='...')
    # execution = SMKLiveExecutionManager(broker)
    
    # 4. Load data and run
    # bars = load_data()
    # pipeline.load_bars(bars)
    
    # Simulate
    for i in range(100):
        result = pipeline.step()
        if result:
            # Get λ scores
            lambda7_score = lambda7.correlation_engine.update(result['bar']['close'], 105.0)
            lambda8_score = 0.2  # from detector
            
            # Execute if signal present
            # await execution.on_smk_signal(
            #     smk_result=result,
            #     lambda7_score=lambda7_score,
            #     lambda8_score=lambda8_score,
            #     fusion_p=result['fusion']['p_fused'],
            #     current_price=result['bar']['close']
            # )
    
    print("Order management system ready!")


if __name__ == "__main__":
    asyncio.run(main())
