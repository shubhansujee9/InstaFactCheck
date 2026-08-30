// Basic widget test for InstaFactCheck.

import 'package:flutter_test/flutter_test.dart';
import 'package:insta_fact_check/main.dart';

void main() {
  testWidgets('App renders home screen', (WidgetTester tester) async {
    await tester.pumpWidget(const InstaFactCheckApp());
    expect(find.text('InstaFactCheck'), findsOneWidget);
    expect(find.text('Fact-Check a Reel'), findsOneWidget);
  });
}
